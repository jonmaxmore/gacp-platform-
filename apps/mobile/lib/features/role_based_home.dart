import 'package:flutter/material.dart';
import 'package:gacp_mobile/features/farmer/dashboard.dart';

class RoleBasedHome extends StatefulWidget {
  const RoleBasedHome({super.key});

  @override
  State<RoleBasedHome> createState() => _RoleBasedHomeState();
}

class _RoleBasedHomeState extends State<RoleBasedHome> {
  String _userRole = 'farmer'; // farmer, dpm_officer, processor, admin
  int _selectedIndex = 0;

  List<Widget> get _screens {
    switch (_userRole) {
      case 'farmer':
        return [
          const FarmerDashboard(),
          Container(), // GACP Certification placeholder
          Container(), // Crop Management placeholder
          Container(), // Traceability placeholder
          Container(), // Profile placeholder
        ];
      case 'dpm_officer':
        return [
          Container(), // Certification Dashboard placeholder
          Container(), // Field Assessment placeholder
          Container(), // Herb Database placeholder
          Container(), // Analytics placeholder
          Container(), // Profile placeholder
        ];
      case 'admin':
        return [
          Container(), // System Dashboard placeholder
          Container(), // User Management placeholder
          Container(), // IoT Monitoring placeholder
          Container(), // AI Models placeholder
          Container(), // Settings placeholder
        ];
      default:
        return [const Center(child: Text('Unknown Role'))];
    }
  }

  List<BottomNavigationBarItem> get _navItems {
    switch (_userRole) {
      case 'farmer':
        return const [
          BottomNavigationBarItem(icon: Icon(Icons.dashboard), label: 'Dashboard'),
          BottomNavigationBarItem(icon: Icon(Icons.verified_user), label: 'GACP'),
          BottomNavigationBarItem(icon: Icon(Icons.spa), label: 'Crops'),
          BottomNavigationBarItem(icon: Icon(Icons.qr_code), label: 'Trace'),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Profile'),
        ];
      case 'dpm_officer':
        return const [
          BottomNavigationBarItem(icon: Icon(Icons.assignment), label: 'Certifications'),
          BottomNavigationBarItem(icon: Icon(Icons.agriculture), label: 'Field'),
          BottomNavigationBarItem(icon: Icon(Icons.local_florist), label: 'Herbs'),
          BottomNavigationBarItem(icon: Icon(Icons.analytics), label: 'Analytics'),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Profile'),
        ];
      case 'admin':
        return const [
          BottomNavigationBarItem(icon: Icon(Icons.dashboard), label: 'System'),
          BottomNavigationBarItem(icon: Icon(Icons.people), label: 'Users'),
          BottomNavigationBarItem(icon: Icon(Icons.sensors), label: 'IoT'),
          BottomNavigationBarItem(icon: Icon(Icons.psychology), label: 'AI'),
          BottomNavigationBarItem(icon: Icon(Icons.settings), label: 'Settings'),
        ];
      default:
        return [];
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('GACP Platform'),
        actions: [
          IconButton(icon: const Icon(Icons.notifications), onPressed: () {}),
          PopupMenuButton(
            itemBuilder: (context) => [
              const PopupMenuItem(value: 'farmer', child: Text('เกษตรกร')),
              const PopupMenuItem(value: 'dpm_officer', child: Text('เจ้าหน้าที่กรม')),
              const PopupMenuItem(value: 'admin', child: Text('ผู้ดูแลระบบ')),
            ],
            onSelected: (value) => setState(() => _userRole = value),
          ),
        ],
      ),
      body: _screens[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: (index) => setState(() => _selectedIndex = index),
        type: BottomNavigationBarType.fixed,
        items: _navItems,
      ),
    );
  }
}