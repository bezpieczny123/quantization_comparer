import 'dart:io';
import 'dart:typed_data';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class Classifier {
  Interpreter? _interpreter;
  List<String>? _labels;

  final int _inputSize = 224;

  TensorType _inputType = TensorType.float32;
  TensorType _outputType = TensorType.float32;

  Future<void> loadModel({required String modelFile, required bool useGpu}) async {
    try {
      if (_interpreter != null) {
        _interpreter!.close();
        _interpreter = null;
      }

      final options = InterpreterOptions();

      // Integer models run better on CPU generally
      bool forceCpu = modelFile.contains("integer");

      if (!useGpu || forceCpu) {
        options.threads = 4;
      } else if (Platform.isAndroid) {
        try {
          options.addDelegate(GpuDelegateV2());
        } catch (e) {
          print("Error GPU Delegate: $e");
        }
      }

      _interpreter = await Interpreter.fromAsset(modelFile, options: options);
      _interpreter!.allocateTensors();
      await _loadLabels();

      try {
        final inputTensor = _interpreter!.getInputTensor(0);
        final outputTensor = _interpreter!.getOutputTensor(0);
        _inputType = inputTensor.type;
        _outputType = outputTensor.type;
        print("Model loaded: $modelFile");
        print("Input Params: Scale=${inputTensor.params.scale},"
            "ZP=${inputTensor.params.zeroPoint}");
      } catch (e) {
        print("Warning: Could not read tensor types: $e");
      }

    } catch (e) {
      print("CRITICAL ERROR LOADING MODEL: $e");
    }
  }

  Future<void> _loadLabels() async {
    try {
      final labelData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelData.split('\n')
          .map((e) => e.trim())
          .where((item) => item.isNotEmpty)
          .toList();
    } catch (e) {
      print("Error loading labels: $e");
    }
  }

  List<double> _softmax(List<double> logits) {
    if (logits.isEmpty) return [];
    double maxLogit = logits.reduce(max);
    List<double> exps = logits.map((x) => exp(x - maxLogit)).toList();
    double sumExps = exps.reduce((curr, next) => curr + next);
    return exps.map((x) => x / sumExps).toList();
  }

  Map<String, dynamic>? predict(img.Image image) {
    if (_interpreter == null) return null;

    try {
      // PREPROCESSING (Excluded from timing)
      img.Image processed = img.copyRotate(image, angle: 90);
      int size = min(processed.width, processed.height);
      processed = img.copyCrop(
          processed,
          x: (processed.width - size) ~/ 2,
          y: (processed.height - size) ~/ 2,
          width: size,
          height: size
      );
      processed = img.copyResize(processed, width: _inputSize, height: _inputSize);

      final stopwatch = Stopwatch()..start();

      final inputTensor = _interpreter!.getInputTensor(0);
      Object inputBuffer;

      if (_inputType == TensorType.float32) {
        final inputBytes = Float32List(1 * _inputSize * _inputSize * 3);
        var pixelIndex = 0;
        for (var y = 0; y < _inputSize; y++) {
          for (var x = 0; x < _inputSize; x++) {
            final pixel = processed.getPixel(x, y);
            inputBytes[pixelIndex++] = pixel.r / 1.0;
            inputBytes[pixelIndex++] = pixel.g / 1.0;
            inputBytes[pixelIndex++] = pixel.b / 1.0;
          }
        }
        inputBuffer = inputBytes.reshape([1, _inputSize, _inputSize, 3]);

      } else {
        // INT8 / UINT8 QUANTIZATION HANDLING
        final params = inputTensor.params;
        final double scale = params.scale;
        final int zeroPoint = params.zeroPoint;

        final bool isUint8 = _inputType == TensorType.uint8;
        final inputBytes = isUint8
            ? Uint8List(1 * _inputSize * _inputSize * 3)
            : Int8List(1 * _inputSize * _inputSize * 3);

        var pixelIndex = 0;
        for (var y = 0; y < _inputSize; y++) {
          for (var x = 0; x < _inputSize; x++) {
            final pixel = processed.getPixel(x, y);

            double rNorm = pixel.r / 1.0;
            double gNorm = pixel.g / 1.0;
            double bNorm = pixel.b / 1.0;

            // Apply Formula: q = (r / scale) + zeroPoint
            int rQ = (rNorm / scale + zeroPoint).round();
            int gQ = (gNorm / scale + zeroPoint).round();
            int bQ = (bNorm / scale + zeroPoint).round();

            // Clamp values to valid range
            if (isUint8) {
              inputBytes[pixelIndex++] = rQ.clamp(0, 255);
              inputBytes[pixelIndex++] = gQ.clamp(0, 255);
              inputBytes[pixelIndex++] = bQ.clamp(0, 255);
            } else {
              inputBytes[pixelIndex++] = rQ.clamp(-128, 127);
              inputBytes[pixelIndex++] = gQ.clamp(-128, 127);
              inputBytes[pixelIndex++] = bQ.clamp(-128, 127);
            }
          }
        }
        inputBuffer = inputBytes.reshape([1, _inputSize, _inputSize, 3]);
      }

      final outputTensor = _interpreter!.getOutputTensor(0);
      final int numClasses = outputTensor.shape[1];
      List<double> finalLogits;

      if (_outputType == TensorType.float32) {
        final outputBuffer = List.filled(numClasses, 0.0).reshape([1, numClasses]);
        _interpreter!.run(inputBuffer, outputBuffer);
        finalLogits = List<double>.from(outputBuffer[0]);
      } else {
        final outputBuffer = List.filled(numClasses, 0).reshape([1, numClasses]);
        _interpreter!.run(inputBuffer, outputBuffer);
        final rawInts = List<int>.from(outputBuffer[0]);

        // Output Dequantization
        final params = outputTensor.params;
        final double scale = params.scale;
        final int zeroPoint = params.zeroPoint;

        if (scale > 0) {
          finalLogits = rawInts.map((e) => (e - zeroPoint) * scale).toList();
        } else {
          finalLogits = rawInts.map((e) => e.toDouble()).toList();
        }
      }

      stopwatch.stop();

      final probabilities = _softmax(finalLogits);
      double maxScore = -1.0;
      int maxIndex = -1;

      for (var i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxScore) {
          maxScore = probabilities[i];
          maxIndex = i;
        }
      }

      return {
        'label': _labels != null && maxIndex != -1 ? _labels![maxIndex] : "Unknown",
        'confidence': maxScore,
        'inferenceTime': stopwatch.elapsedMilliseconds,
      };

    } catch (e) {
      print("Error during prediction: $e");
      return null;
    }
  }
}