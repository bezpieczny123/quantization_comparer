import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

/// Helper class to store a copy of the camera frame data.
class CameraFrameData {
  final Uint8List yPlane;
  final Uint8List uPlane;
  final Uint8List vPlane;
  final int bytesPerRow;
  final int uvRowStride;
  final int uvPixelStride;
  final int height;
  final int width;

  CameraFrameData({
    required this.yPlane,
    required this.uPlane,
    required this.vPlane,
    required this.bytesPerRow,
    required this.uvRowStride,
    required this.uvPixelStride,
    required this.height,
    required this.width,
  });

  static CameraFrameData fromCameraImage(CameraImage image) {
    return CameraFrameData(
      yPlane: Uint8List.fromList(image.planes[0].bytes),
      uPlane: Uint8List.fromList(image.planes[1].bytes),
      vPlane: Uint8List.fromList(image.planes[2].bytes),
      bytesPerRow: image.planes[0].bytesPerRow,
      uvRowStride: image.planes[1].bytesPerRow,
      uvPixelStride: image.planes[1].bytesPerPixel ?? 1,
      height: image.height,
      width: image.width,
    );
  }
}

img.Image? convertYUV420ToImage(CameraFrameData data) {
  final int width = data.width;
  final int height = data.height;

  final img.Image image = img.Image(width: width, height: height);

  final bytesY = data.yPlane;
  final bytesU = data.uPlane;
  final bytesV = data.vPlane;

  final int yRowStride = data.bytesPerRow;
  final int uvRowStride = data.uvRowStride;
  final int uvPixelStride = data.uvPixelStride;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final int yIndex = (y * yRowStride) + x;
      final int uvIndex = (uvPixelStride * (x / 2).floor()) + (uvRowStride * (y / 2).floor());

      if (yIndex >= bytesY.length || uvIndex >= bytesU.length || uvIndex >= bytesV.length) {
        continue;
      }

      final int yp = bytesY[yIndex] & 0xFF;
      final int up = bytesU[uvIndex] & 0xFF;
      final int vp = bytesV[uvIndex] & 0xFF;

      int r = (yp + 1.402 * (vp - 128)).toInt();
      int g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128)).toInt();
      int b = (yp + 1.772 * (up - 128)).toInt();

      r = r.clamp(0, 255);
      g = g.clamp(0, 255);
      b = b.clamp(0, 255);

      image.setPixelRgb(x, y, r, g, b);
    }
  }

  return image;
}