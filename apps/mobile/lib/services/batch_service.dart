class BatchService {
  final FirebaseFirestore _firestore = FirebaseService().firestore;
  final FirebaseStorage _storage = FirebaseService().storage;

  // Create a new batch
  Future<void> createBatch({
    required String herbId,
    required String farmId,
    required DateTime plantingDate,
    required int quantity,
  }) async {
    DocumentReference docRef = await _firestore.collection('batches').add({
      'herbId': herbId,
      'farmId': farmId,
      'plantingDate': Timestamp.fromDate(plantingDate),
      'quantity': quantity,
      'qualityScore': 0,
      'locations': [
        {
          'type': 'farm',
          'timestamp': FieldValue.serverTimestamp(),
        }
      ],
      'createdAt': FieldValue.serverTimestamp(),
      'updatedAt': FieldValue.serverTimestamp(),
    });

    // Generate QR code URL
    String qrCodeUrl = await generateQRCode(docRef.id);
    
    // Update batch with QR code
    await docRef.update({
      'qrCode': qrCodeUrl,
    });
  }

  // Generate QR code for batch
  Future<String> generateQRCode(String batchId) async {
    // Implementation depends on your QR generation method
    // This could be an API call or local generation
    return 'https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=gacp-batch-$batchId';
  }

  // Update batch location
  Future<void> updateBatchLocation(String batchId, String locationType) async {
    await _firestore.collection('batches').doc(batchId).update({
      'locations': FieldValue.arrayUnion([
        {
          'type': locationType,
          'timestamp': FieldValue.serverTimestamp(),
        }
      ]),
      'updatedAt': FieldValue.serverTimestamp(),
    });
  }
}