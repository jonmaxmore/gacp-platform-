import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:syncfusion_flutter_gauges/gauges.dart';

class IoTDashboard extends StatefulWidget {
  final String farmId;

  const IoTDashboard({super.key, required this.farmId});

  @override
  State<IoTDashboard> createState() => _IoTDashboardState();
}

class _IoTDashboardState extends State<IoTDashboard> {
  late Stream<QuerySnapshot> _sensorStream;

  @override
  void initState() {
    super.initState();
    _sensorStream = FirebaseFirestore.instance
        .collection('farms')
        .doc(widget.farmId)
        .collection('sensorData')
        .orderBy('timestamp', descending: true)
        .limit(1)
        .snapshots();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Farm Monitoring')),
      body: StreamBuilder<QuerySnapshot>(
        stream: _sensorStream,
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          }

          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }

          if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
            return const Center(child: Text('No sensor data available'));
          }

          final data = snapshot.data!.docs.first.data() as Map<String, dynamic>;
          final temperature = data['temperature']?.toDouble() ?? 0.0;
          final humidity = data['humidity']?.toDouble() ?? 0.0;
          final soilMoisture = data['soilMoisture']?.toInt() ?? 0;
          final light = data['light']?.toDouble() ?? 0.0;

          return Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: [
                Row(
                  children: [
                    _buildGauge(
                      title: 'Temperature',
                      value: temperature,
                      unit: '°C',
                      min: 0,
                      max: 50,
                    ),
                    _buildGauge(
                      title: 'Humidity',
                      value: humidity,
                      unit: '%',
                      min: 0,
                      max: 100,
                    ),
                  ],
                ),
                const SizedBox(height: 20),
                Row(
                  children: [
                    _buildGauge(
                      title: 'Soil Moisture',
                      value: soilMoisture.toDouble(),
                      unit: '',
                      min: 0,
                      max: 4095,
                    ),
                    _buildGauge(
                      title: 'Light',
                      value: light,
                      unit: 'lux',
                      min: 0,
                      max: 1000,
                    ),
                  ],
                ),
                const SizedBox(height: 20),
                _buildHistoryChart(widget.farmId),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildGauge({required String title, required double value, required String unit, double min = 0, double max = 100}) {
    return Expanded(
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              Text(title, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              const SizedBox(height: 10),
              SizedBox(
                height: 150,
                child: SfRadialGauge(
                  axes: <RadialAxis>[
                    RadialAxis(
                      minimum: min,
                      maximum: max,
                      ranges: <GaugeRange>[
                        GaugeRange(
                          startValue: min,
                          endValue: max,
                          color: _getRangeColor(title, value),
                        ),
                      ],
                      pointers: <GaugePointer>[
                        NeedlePointer(
                          value: value,
                          enableAnimation: true,
                        ),
                      ],
                      annotations: <GaugeAnnotation>[
                        GaugeAnnotation(
                          widget: Text(
                            '${value.toStringAsFixed(1)} $unit',
                            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                          ),
                          positionFactor: 0.5,
                          angle: 90,
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Color _getRangeColor(String title, double value) {
    if (title == 'Temperature') {
      if (value < 10) return Colors.blue[100]!;
      if (value > 30) return Colors.red[100]!;
      return Colors.green[100]!;
    } else if (title == 'Humidity') {
      if (value < 30) return Colors.orange[100]!;
      if (value > 70) return Colors.blue[100]!;
      return Colors.green[100]!;
    } else if (title == 'Soil Moisture') {
      if (value < 1000) return Colors.red[100]!;
      if (value > 3000) return Colors.blue[100]!;
      return Colors.green[100]!;
    } else {
      return Colors.grey[100]!;
    }
  }

  Widget _buildHistoryChart(String farmId) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Historical Data', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            SizedBox(
              height: 200,
              child: StreamBuilder<QuerySnapshot>(
                stream: FirebaseFirestore.instance
                    .collection('farms')
                    .doc(farmId)
                    .collection('sensorData')
                    .orderBy('timestamp', descending: true)
                    .limit(20)
                    .snapshots(),
                builder: (context, snapshot) {
                  if (!snapshot.hasData) return const Center(child: CircularProgressIndicator());
                  
                  final docs = snapshot.data!.docs.reversed.toList();
                  
                  return SfCartesianChart(
                    primaryXAxis: DateTimeAxis(
                      title: AxisTitle(text: 'Time'),
                      intervalType: DateTimeIntervalType.minutes,
                    ),
                    series: <ChartSeries>[
                      LineSeries<Map<String, dynamic>, DateTime>(
                        dataSource: docs,
                        xValueMapper: (data, _) => (data['timestamp'] as Timestamp).toDate(),
                        yValueMapper: (data, _) => data['temperature'],
                        name: 'Temperature',
                        markerSettings: const MarkerSettings(isVisible: true),
                      ),
                      LineSeries<Map<String, dynamic>, DateTime>(
                        dataSource: docs,
                        xValueMapper: (data, _) => (data['timestamp'] as Timestamp).toDate(),
                        yValueMapper: (data, _) => data['humidity'],
                        name: 'Humidity',
                        markerSettings: const MarkerSettings(isVisible: true),
                      ),
                    ],
                    legend: Legend(isVisible: true),
                    tooltipBehavior: TooltipBehavior(enable: true),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}