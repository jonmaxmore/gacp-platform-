import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class AIService {
  static const String _baseUrl = 'https://api.gacp-platform.com';
  static const String _modelPath = 'assets/models/herb_detection_model.tflite';
  
  Interpreter? _interpreter;
  bool _modelLoaded = false;

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(_modelPath);
      _modelLoaded = true;
      print('AI Model loaded successfully');
    } catch (e) {
      print('Error loading AI model: $e');
      throw Exception('Failed to load AI model');
    }
  }

  Future<PredictionResult> predictHerbQuality({
    required File imageFile,
    required String herbType,
    bool useOnlineModel = true,
  }) async {
    if (useOnlineModel && await _hasInternetConnection()) {
      return await _predictOnline(imageFile, herbType);
    } else {
      return await _predictOffline(imageFile, herbType);
    }
  }

  Future<PredictionResult> _predictOnline(File imageFile, String herbType) async {
    try {
      final bytes = await imageFile.readAsBytes();
      final base64Image = base64Encode(bytes);

      final response = await http.post(
        Uri.parse('$_baseUrl/api/ai/predict'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ${await _getAuthToken()}',
        },
        body: jsonEncode({
          'image': base64Image,
          'herb_type': herbType,
          'model_version': 'latest',
          'use_confidence_scoring': true,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return PredictionResult.fromJson(data);
      } else {
        throw Exception('Online prediction failed: ${response.statusCode}');
      }
    } catch (e) {
      print('Online prediction error: $e');
      // Fallback to offline prediction
      return await _predictOffline(imageFile, herbType);
    }
  }

  Future<PredictionResult> _predictOffline(File imageFile, String herbType) async {
    if (!_modelLoaded) {
      await loadModel();
    }

    try {
      // Preprocess image
      final inputImage = await _preprocessImage(imageFile);
      
      // Prepare input tensor
      final input = [inputImage];
      
      // Prepare output tensor
      final output = List.generate(1, (index) => List.filled(6, 0.0));
      
      // Run inference
      _interpreter!.run(input, output);
      
      // Process results
      final predictions = output[0];
      final maxIndex = _getMaxIndex(predictions);
      final confidence = predictions[maxIndex];
      
      return PredictionResult(
        herbType: herbType,
        qualityGrade: _getQualityGrade(maxIndex),
        confidence: confidence,
        predictions: _formatPredictions(predictions),
        isOffline: true,
        timestamp: DateTime.now(),
      );
    } catch (e) {
      print('Offline prediction error: $e');
      throw Exception('Prediction failed: $e');
    }
  }

  Future<List<List<double>>> _preprocessImage(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    final image = img.decodeImage(bytes);
    
    if (image == null) {
      throw Exception('Unable to decode image');
    }

    // Resize to model input size (224x224)
    final resized = img.copyResize(image, width: 224, height: 224);
    
    // Convert to normalized float values
    final input = List.generate(224, (y) =>
      List.generate(224, (x) =>
        List.generate(3, (c) {
          final pixel = resized.getPixel(x, y);
          switch (c) {
            case 0: return (img.getRed(pixel) / 255.0 - 0.485) / 0.229;
            case 1: return (img.getGreen(pixel) / 255.0 - 0.456) / 0.224;
            case 2: return (img.getBlue(pixel) / 255.0 - 0.406) / 0.225;
            default: return 0.0;
          }
        })
      )
    );

    return input;
  }

  int _getMaxIndex(List<double> predictions) {
    double max = predictions[0];
    int maxIndex = 0;
    
    for (int i = 1; i < predictions.length; i++) {
      if (predictions[i] > max) {
        max = predictions[i];
        maxIndex = i;
      }
    }
    
    return maxIndex;
  }

  String _getQualityGrade(int index) {
    const grades = ['Grade A', 'Grade B', 'Grade C', 'Grade D', 'Rejected', 'Needs Review'];
    return grades[index];
  }

  Map<String, double> _formatPredictions(List<double> predictions) {
    const labels = ['Grade A', 'Grade B', 'Grade C', 'Grade D', 'Rejected', 'Needs Review'];
    final Map<String, double> result = {};
    
    for (int i = 0; i < predictions.length; i++) {
      result[labels[i]] = predictions[i];
    }
    
    return result;
  }

  Future<bool> _hasInternetConnection() async {
    try {
      final result = await http.get(Uri.parse('$_baseUrl/health'));
      return result.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  Future<String> _getAuthToken() async {
    // Implement your authentication logic here
    return 'your_auth_token';
  }

  Future<List<ABTestVariant>> getABTestVariants(String userId) async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl/api/ab-testing/variants?user_id=$userId'),
        headers: {
          'Authorization': 'Bearer ${await _getAuthToken()}',
        },
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return (data['variants'] as List)
            .map((variant) => ABTestVariant.fromJson(variant))
            .toList();
      } else {
        throw Exception('Failed to fetch AB test variants');
      }
    } catch (e) {
      print('Error fetching AB test variants: $e');
      return [];
    }
  }

  Future<void> trackABTestEvent({
    required String experimentId,
    required String variant,
    required String userId,
    required String eventType,
    Map<String, dynamic>? metadata,
  }) async {
    try {
      await http.post(
        Uri.parse('$_baseUrl/api/ab-testing/track'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ${await _getAuthToken()}',
        },
        body: jsonEncode({
          'experiment_id': experimentId,
          'variant': variant,
          'user_id': userId,
          'event_type': eventType,
          'metadata': metadata,
          'timestamp': DateTime.now().millisecondsSinceEpoch,
        }),
      );
    } catch (e) {
      print('Error tracking AB test event: $e');
    }
  }
}

class PredictionResult {
  final String herbType;
  final String qualityGrade;
  final double confidence;
  final Map<String, double> predictions;
  final bool isOffline;
  final DateTime timestamp;

  PredictionResult({
    required this.herbType,
    required this.qualityGrade,
    required this.confidence,
    required this.predictions,
    required this.isOffline,
    required this.timestamp,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      herbType: json['herb_type'],
      qualityGrade: json['quality_grade'],
      confidence: json['confidence'].toDouble(),
      predictions: Map<String, double>.from(json['predictions']),
      isOffline: json['is_offline'] ?? false,
      timestamp: DateTime.parse(json['timestamp']),
    );
  }
}

class ABTestVariant {
  final String experimentId;
  final String variant;
  final Map<String, dynamic> config;

  ABTestVariant({
    required this.experimentId,
    required this.variant,
    required this.config,
  });

  factory ABTestVariant.fromJson(Map<String, dynamic> json) {
    return ABTestVariant(
      experimentId: json['experiment_id'],
      variant: json['variant'],
      config: json['config'],
    );
  }
}