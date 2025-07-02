import 'dart:io';
import 'dart:typed_data';
import 'dart:math';
import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/material.dart';

class DiseaseDetector {
  late Interpreter _interpreter;
  late List<String> _labels;
  late Map<String, dynamic> _modelInfo;
  bool _isInitialized = false;

  Future<void> initialize() async {
    if (_isInitialized) return;

    // Load model
    _interpreter = await Interpreter.fromAsset('ai-models/disease_detection/efficientnet_v2.tflite');
    
    // Load labels
    final labelData = await rootBundle.loadString('ai-models/disease_detection/labels.txt');
    _labels = labelData.split('\n');
    
    // Load model info
    final versionData = await rootBundle.loadString('ai-models/disease_detection/version.json');
    _modelInfo = jsonDecode(versionData);
    
    _isInitialized = true;
  }

  Future<Map<String, dynamic>> detectDisease(File imageFile) async {
    if (!_isInitialized) await initialize();

    // Load and preprocess image
    final imageBytes = await imageFile.readAsBytes();
    final image = img.decodeImage(imageBytes)!;
    
    // Apply model-specific preprocessing
    final preprocessedImage = _preprocessImage(image);
    
    // Create input tensor
    final input = _imageToInputTensor(preprocessedImage);
    
    // Prepare output tensor
    final output = List.filled(1 * _labels.length, 0.0).reshape([1, _labels.length]);
    
    // Run inference
    final stopwatch = Stopwatch()..start();
    _interpreter.run(input, output);
    final inferenceTime = stopwatch.elapsedMilliseconds;
    
    // Process results
    final results = _processOutput(output[0]);
    
    return {
      'model': _modelInfo,
      'inference_time': inferenceTime,
      'results': results,
    };
  }

  img.Image _preprocessImage(img.Image image) {
    // Get target size from model info
    final targetSize = _modelInfo['input_size'] ?? 224;
    
    // Apply preprocessing pipeline
    return image
      .. = img.copyResize(image, width: targetSize, height: targetSize)
      .. = _applyNormalization(image);
  }

  img.Image _applyNormalization(img.Image image) {
    // Apply normalization based on model requirements
    final mean = _modelInfo['mean'] ?? [0.485, 0.456, 0.406];
    final std = _modelInfo['std'] ?? [0.229, 0.224, 0.225];
    
    for (var y = 0; y < image.height; y++) {
      for (var x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        
        final r = (img.getRed(pixel) / 255.0 - mean[0]) / std[0];
        final g = (img.getGreen(pixel) / 255.0 - mean[1]) / std[1];
        final b = (img.getBlue(pixel) / 255.0 - mean[2]) / std[2];
        
        image.setPixelRgb(
          x, 
          y, 
          (r * 255).clamp(0, 255).toInt(),
          (g * 255).clamp(0, 255).toInt(),
          (b * 255).clamp(0, 255).toInt(),
        );
      }
    }
    return image;
  }

  Float32List _imageToInputTensor(img.Image image) {
    final targetSize = _modelInfo['input_size'] ?? 224;
    final channels = 3;
    final convertedBytes = Float32List(1 * targetSize * targetSize * channels);
    final buffer = Float32List.view(convertedBytes.buffer);

    int pixelIndex = 0;
    for (int y = 0; y < targetSize; y++) {
      for (int x = 0; x < targetSize; x++) {
        final pixel = image.getPixel(x, y);
        
        // Convert to float32 and normalize
        buffer[pixelIndex++] = img.getRed(pixel) / 255.0;
        buffer[pixelIndex++] = img.getGreen(pixel) / 255.0;
        buffer[pixelIndex++] = img.getBlue(pixel) / 255.0;
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }

  List<Map<String, dynamic>> _processOutput(List<double> probabilities) {
    final results = <Map<String, dynamic>>[];
    
    for (int i = 0; i < probabilities.length; i++) {
      if (_labels[i].trim().isEmpty) continue;
      
      results.add({
        'disease': _labels[i],
        'confidence': probabilities[i],
        'severity': _calculateSeverity(probabilities[i]),
      });
    }
    
    // Sort by confidence
    results.sort((a, b) => b['confidence'].compareTo(a['confidence']));
    
    return results.sublist(0, min(5, results.length));
  }

  String _calculateSeverity(double confidence) {
    if (confidence > 0.8) return 'รุนแรง';
    if (confidence > 0.5) return 'ปานกลาง';
    return 'เล็กน้อย';
  }
}