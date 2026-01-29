import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'classifier.dart';
import 'image_utils.dart';

const List<String> modelFiles = [
  'assets/fifth_float16_quant.tflite',
  'assets/fifth_dynamic_quant.tflite',
  'assets/fifth_integer_quant.tflite',
  'assets/baseline_dynamic_quant.tflite',
  'assets/baseline_float16_quant.tflite',
  'assets/baseline_integer_quant.tflite',
  'assets/forth_dynamic_quant.tflite',
  'assets/forth_float16_quant.tflite',
  'assets/forth_integer_quant.tflite',
  'assets/second_dynamic_quant.tflite',
  'assets/second_float16_quant.tflite',
  'assets/second_integer_quant.tflite',
  'assets/forth_model.tflite',
  'assets/second_model.tflite',
  'assets/baseline_model.tflite',
  'assets/fifth_model.tflite',
];

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MaterialApp(home: CameraClassifierScreen()));
}

class CameraClassifierScreen extends StatefulWidget {
  const CameraClassifierScreen({super.key});

  @override
  State<CameraClassifierScreen> createState() => _CameraClassifierScreenState();
}

enum BenchmarkState { idle, warmup, measuring }

class _CameraClassifierScreenState extends State<CameraClassifierScreen> {
  CameraController? _controller;
  final Classifier _classifier = Classifier();

  // Processing state
  bool _isProcessing = false;
  int _frameSkipCounter = 0;

  // Live Results
  String _resultLabel = "Initializing...";
  String _resultConf = "";
  double _avgInferenceTime = 0.0;

  // Settings
  String _selectedModel = modelFiles[0];
  bool _useGpu = false;

  // Benchmark State
  bool _isBenchmarking = false;
  BenchmarkState _benchmarkState = BenchmarkState.idle;
  int _benchmarkModelIndex = 0;
  DateTime? _phaseStartTime;
  final List<double> _currentSamples = [];
  final Map<String, double> _benchmarkResults = {};

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    await Permission.camera.request();
    final cameras = await availableCameras();

    await _classifier.loadModel(modelFile: _selectedModel, useGpu: _useGpu);

    _controller = CameraController(
      cameras[0],
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await _controller!.initialize();
    _controller!.startImageStream(_processFrame);

    if (mounted) setState(() {});
  }

  Future<void> _processFrame(CameraImage cameraImage) async {
    if (_isProcessing) return;

    // Throttling: Always skip frames to prevent UI freeze,
    // but process enough to get good benchmark data.
    _frameSkipCounter++;
    if (_frameSkipCounter % 3 != 0) return;

    _isProcessing = true;

    // Fast Copy & Release Buffer
    final frameData = CameraFrameData.fromCameraImage(cameraImage);

    _runInferenceBackground(frameData);
  }

  void _runInferenceBackground(CameraFrameData frameData) async {
    try {
      // 1. Convert YUV to RGB in background
      final convertedImage = await compute(convertYUV420ToImage, frameData);

      if (convertedImage != null) {
        // 2. Run Prediction
        final result = _classifier.predict(convertedImage);

        if (mounted && result != null) {
          if (_isBenchmarking) {
            _handleBenchmarkLogic(result);
          } else {
            _updateLiveStats(result);
          }
        }
      }
    } catch (e) {
      print("Error processing frame: $e");
    } finally {
      _isProcessing = false;
    }
  }

  void _updateLiveStats(Map<String, dynamic> result) {
    final int inferenceMs = result['inferenceTime'] ?? 0;

    if (_avgInferenceTime == 0.0) {
      _avgInferenceTime = inferenceMs.toDouble();
    } else {
      _avgInferenceTime = (_avgInferenceTime * 0.8) + (inferenceMs * 0.2);
    }

    setState(() {
      _resultLabel = result['label'];
      _resultConf = "${(result['confidence'] * 100).toStringAsFixed(1)}%";
    });
  }

  // --- BENCHMARKING LOGIC ---

  void _startBenchmark() async {
    await _controller?.stopImageStream();

    setState(() {
      _isBenchmarking = true;
      _benchmarkResults.clear();
      _benchmarkModelIndex = 0;
      _benchmarkState = BenchmarkState.warmup;
      _phaseStartTime = DateTime.now();
      _currentSamples.clear();
    });

    // Load first model
    await _classifier.loadModel(modelFile: modelFiles[0], useGpu: _useGpu);

    await _controller?.startImageStream(_processFrame);
  }

  void _handleBenchmarkLogic(Map<String, dynamic> result) async {
    final now = DateTime.now();
    final double inferenceTime = (result['inferenceTime'] as int).toDouble();

    if (_benchmarkState == BenchmarkState.warmup) {
      // Warmup Phase (5 seconds)
      if (now.difference(_phaseStartTime!).inSeconds >= 5) {
        // Switch to Measuring
        _phaseStartTime = now;
        _benchmarkState = BenchmarkState.measuring;
        _currentSamples.clear();
        print("Benchmark: Finished warmup for ${modelFiles[_benchmarkModelIndex]}");
      }
    } else if (_benchmarkState == BenchmarkState.measuring) {
      // Measuring Phase (20 seconds)
      _currentSamples.add(inferenceTime);

      if (now.difference(_phaseStartTime!).inSeconds >= 20) {
        // Finish this model
        double avg = _currentSamples.isEmpty
            ? 0.0
            : _currentSamples.reduce((a, b) => a + b) / _currentSamples.length;

        _benchmarkResults[modelFiles[_benchmarkModelIndex]] = avg;
        print("Benchmark: Finished ${modelFiles[_benchmarkModelIndex]} -> $avg ms");

        // Move to next model?
        if (_benchmarkModelIndex < modelFiles.length - 1) {
          await _controller?.stopImageStream();

          _benchmarkModelIndex++;
          await _classifier.loadModel(modelFile: modelFiles[_benchmarkModelIndex], useGpu: _useGpu);

          // Reset for next model
          _phaseStartTime = DateTime.now();
          _benchmarkState = BenchmarkState.warmup;
          _currentSamples.clear();

          await _controller?.startImageStream(_processFrame);
        } else {
          _finishBenchmark();
        }
      }
    }

    // Force UI rebuild to show progress
    setState(() {});
  }

  void _finishBenchmark() {
    _isBenchmarking = false;
    _benchmarkState = BenchmarkState.idle;
    _showBenchmarkResults();

    // Reload the user's originally selected model
    _onModelChanged(_selectedModel);
  }

  void _showBenchmarkResults() {
    // Sort results by time (fastest first)
    final sortedEntries = _benchmarkResults.entries.toList()
      ..sort((a, b) => a.value.compareTo(b.value));

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("Benchmark Results"),
        content: SizedBox(
          width: double.maxFinite,
          height: 400,
          child: ListView.builder(
            itemCount: sortedEntries.length,
            itemBuilder: (ctx, index) {
              final entry = sortedEntries[index];
              final name = entry.key.split('/').last;
              return ListTile(
                dense: true,
                title: Text(name, style: const TextStyle(fontWeight: FontWeight.bold)),
                trailing: Text("${entry.value.toStringAsFixed(2)} ms"),
              );
            },
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text("Close"))
        ],
      ),
    );
  }

  void _onModelChanged(String? newValue) async {
    if (newValue != null) {
      await _controller?.stopImageStream();
      setState(() {
        _selectedModel = newValue;
        _avgInferenceTime = 0.0;
      });
      await _classifier.loadModel(modelFile: _selectedModel, useGpu: _useGpu);
      _controller?.startImageStream(_processFrame);
    }
  }

  void _onGpuChanged(bool value) async {
    await _controller?.stopImageStream();
    setState(() {
      _useGpu = value;
      _avgInferenceTime = 0.0;
    });
    await _classifier.loadModel(modelFile: _selectedModel, useGpu: _useGpu);
    _controller?.startImageStream(_processFrame);
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(title: const Text("TFLite Classifier")),
      body: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                // Camera Preview
                Positioned.fill(
                  child: AspectRatio(
                    aspectRatio: _controller!.value.aspectRatio,
                    child: CameraPreview(_controller!),
                  ),
                ),

                // Normal Mode Overlay
                if (!_isBenchmarking)
                  Positioned(
                    bottom: 20,
                    left: 20,
                    right: 20,
                    child: Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Column(
                        children: [
                          Text(
                            _resultLabel,
                            style: const TextStyle(
                                color: Colors.white,
                                fontSize: 32,
                                fontWeight: FontWeight.bold
                            ),
                            textAlign: TextAlign.center,
                          ),
                          Text(
                            "Confidence: $_resultConf",
                            style: const TextStyle(color: Colors.yellowAccent, fontSize: 18),
                          ),
                        ],
                      ),
                    ),
                  ),

                // Benchmark Progress Overlay
                if (_isBenchmarking)
                  Positioned.fill(
                    child: Container(
                      color: Colors.black87,
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const CircularProgressIndicator(),
                          const SizedBox(height: 20),
                          Text(
                            "Benchmarking Model ${_benchmarkModelIndex + 1} / ${modelFiles.length}",
                            style: const TextStyle(color: Colors.white, fontSize: 18),
                          ),
                          const SizedBox(height: 10),
                          Text(
                            modelFiles[_benchmarkModelIndex].split('/').last,
                            style: const TextStyle(color: Colors.white70, fontSize: 14),
                          ),
                          const SizedBox(height: 20),
                          Text(
                            _benchmarkState == BenchmarkState.warmup
                                ? "Warming up (5s)..."
                                : "Measuring (20s)...",
                            style: TextStyle(
                                color: _benchmarkState == BenchmarkState.warmup
                                    ? Colors.orange
                                    : Colors.green,
                                fontWeight: FontWeight.bold
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ),

          // Bottom Control Panel (Hidden during benchmark)
          if (!_isBenchmarking)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              color: Colors.grey[200],
              child: Column(
                children: [
                  Text(
                    "Avg Inference: ${_avgInferenceTime.toStringAsFixed(1)} ms",
                    style: const TextStyle(color: Colors.blue, fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  const Divider(),
                  Row(
                    children: [
                      Expanded(
                        child: DropdownButton<String>(
                          isExpanded: true,
                          value: _selectedModel,
                          items: modelFiles.map((v) => DropdownMenuItem(value: v, child: Text(v.split('/').last))).toList(),
                          onChanged: _onModelChanged,
                        ),
                      ),
                      const SizedBox(width: 10),
                      Column(
                        children: [
                          const Text("GPU", style: TextStyle(fontSize: 10)),
                          Switch(
                            value: _useGpu,
                            onChanged: _onGpuChanged,
                          ),
                        ],
                      )
                    ],
                  ),
                  const SizedBox(height: 8),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      icon: const Icon(Icons.speed),
                      label: const Text("RUN BENCHMARK (ALL MODELS)"),
                      style: ElevatedButton.styleFrom(backgroundColor: Colors.blueAccent, foregroundColor: Colors.white),
                      onPressed: _startBenchmark,
                    ),
                  )
                ],
              ),
            ),
        ],
      ),
    );
  }
}