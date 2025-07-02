import 'package:flutter/material.dart';
import 'package:gacp_platform/ai/disease_detection_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'GACP Plant Doctor',
      theme: ThemeData(
        primarySwatch: Colors.green,
      ),
      home: const DiseaseDetectionScreen(),
    );
  }
}