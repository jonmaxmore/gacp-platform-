import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:gacp_platform/ai/disease_detector.dart';

class DiseaseDetectionScreen extends StatefulWidget {
  const DiseaseDetectionScreen({super.key});

  @override
  State<DiseaseDetectionScreen> createState() => _DiseaseDetectionScreenState();
}

class _DiseaseDetectionScreenState extends State<DiseaseDetectionScreen> {
  final DiseaseDetector _detector = DiseaseDetector();
  File? _selectedImage;
  Map<String, dynamic>? _results;
  bool _isProcessing = false;
  String _statusMessage = '';

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
        _results = null;
        _isProcessing = true;
        _statusMessage = 'กำลังวิเคราะห์ภาพ...';
      });

      try {
        final detectionResults = await _detector.detectDisease(_selectedImage!);
        
        setState(() {
          _results = detectionResults;
          _isProcessing = false;
          _statusMessage = 'วิเคราะห์สำเร็จ!';
        });
      } catch (e) {
        setState(() {
          _isProcessing = false;
          _statusMessage = 'เกิดข้อผิดพลาด: ${e.toString()}';
        });
      }
    }
  }

  Widget _buildModelInfo() {
    if (_results == null) return const SizedBox();
    
    final modelInfo = _results!['model'];
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('โมเดล: ${modelInfo['name'] ?? 'N/A'}', 
                style: const TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Text('เวอร์ชัน: ${modelInfo['version'] ?? 'N/A'}'),
            Text('ความแม่นยำ: ${(modelInfo['accuracy'] ?? 0) * 100}%'),
            Text('วันที่ฝึกโมเดล: ${modelInfo['training_date'] ?? 'N/A'}'),
            Text('เวลาในการวิเคราะห์: ${_results!['inference_time']}ms'),
          ],
        ),
      ),
    );
  }

  Widget _buildResults() {
    if (_results == null || _isProcessing) return const SizedBox();
    
    final results = _results!['results'] as List<dynamic>;
    final topResult = results.first;
    
    return Column(
      children: [
        const SizedBox(height: 20),
        Text('ผลการวินิจฉัย: ${topResult['disease']}',
            style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
        const SizedBox(height: 10),
        Text('ความมั่นใจ: ${(topResult['confidence'] * 100).toStringAsFixed(1)}%',
            style: TextStyle(
                fontSize: 18,
                color: topResult['confidence'] > 0.7 ? Colors.red : Colors.orange)),
        Text('ระดับความรุนแรง: ${topResult['severity']}'),
        const SizedBox(height: 20),
        const Text('โรคอื่นๆ ที่อาจเป็นไปได้:',
            style: TextStyle(fontWeight: FontWeight.bold)),
        const SizedBox(height: 10),
        ...results.sublist(1).map((result) => ListTile(
              title: Text(result['disease']),
              trailing: Text('${(result['confidence'] * 100).toStringAsFixed(1)}%'),
              subtitle: LinearProgressIndicator(
                value: result['confidence'],
                color: _getSeverityColor(result['severity']),
              ),
            )),
      ],
    );
  }

  Color _getSeverityColor(String severity) {
    switch (severity) {
      case 'รุนแรง':
        return Colors.red;
      case 'ปานกลาง':
        return Colors.orange;
      default:
        return Colors.yellow;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ระบบตรวจสอบโรคพืช'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            // Image preview
            Container(
              height: 300,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(10),
              ),
              child: _selectedImage != null
                  ? Image.file(_selectedImage!, fit: BoxFit.cover)
                  : const Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.photo, size: 60, color: Colors.grey),
                          Text('เลือกรูปเพื่อเริ่มการวิเคราะห์'),
                        ],
                      ),
                    ),
            ),
            
            const SizedBox(height: 20),
            
            // Status indicator
            if (_isProcessing)
              const Column(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 10),
                  Text('กำลังประมวลผล...'),
                ],
              )
            else if (_statusMessage.isNotEmpty)
              Text(_statusMessage,
                  style: TextStyle(
                      color: _statusMessage.contains('เกิดข้อผิดพลาด')
                          ? Colors.red
                          : Colors.green)),
            
            // Model info
            _buildModelInfo(),
            
            // Results
            _buildResults(),
          ],
        ),
      ),
      floatingActionButton: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          FloatingActionButton(
            heroTag: 'camera',
            onPressed: () => _pickImage(ImageSource.camera),
            tooltip: 'ถ่ายภาพใหม่',
            backgroundColor: Colors.green,
            child: const Icon(Icons.camera_alt),
          ),
          const SizedBox(height: 16),
          FloatingActionButton(
            heroTag: 'gallery',
            onPressed: () => _pickImage(ImageSource.gallery),
            tooltip: 'เลือกรูปจากแกลเลอรี่',
            child: const Icon(Icons.photo_library),
          ),
        ],
      ),
    );
  }
}