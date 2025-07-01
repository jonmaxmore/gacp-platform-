import 'package:flutter/material.dart';
import 'package:gacp_mobile/features/role_based_home.dart';

class GACPErpApp extends StatelessWidget {
  const GACPErpApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'GACP Platform',
      theme: ThemeData(
        primarySwatch: Colors.green,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        fontFamily: 'Prompt',
        useMaterial3: true,
      ),
      home: const RoleBasedHome(),
      debugShowCheckedModeBanner: false,
    );
  }
}