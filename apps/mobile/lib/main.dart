import 'package:flutter/material.dart';
import 'package:gacp_mobile/app.dart';
import 'package:gacp_mobile/services/firebase_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize Firebase
  await FirebaseService().initialize();
  
  runApp(const GACPErpApp());
}