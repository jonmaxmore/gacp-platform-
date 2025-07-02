// ในไฟล์ lib/ai/disease_detector.dart
Future<void> initialize() async {
  // โหลดโมเดล
  _interpreter = await Interpreter.fromAsset('ai-models/disease_detection/efficientnet_v2.tflite');
  
  // โหลด labels
  final labelData = await rootBundle.loadString('ai-models/disease_detection/labels.txt');
  _labels = labelData.split('\n');
  
  // โหลดข้อมูลเวอร์ชัน
  final versionData = await rootBundle.loadString('ai-models/disease_detection/version.json');
  _modelInfo = jsonDecode(versionData);
}