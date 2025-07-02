class CertificationService {
  final FirebaseFirestore _firestore = FirebaseService().firestore;
  final FirebaseStorage _storage = FirebaseService().storage;

  // Submit certification request
  Future<void> submitCertification({
    required String farmId,
    required String standard,
    required List<String> documentUrls,
  }) async {
    await _firestore.collection('certifications').add({
      'farmId': farmId,
      'status': 'pending',
      'standard': standard,
      'documents': documentUrls,
      'requestedDate': FieldValue.serverTimestamp(),
      'updatedAt': FieldValue.serverTimestamp(),
    });
  }

  // Upload certification document
  Future<String> uploadCertificationDocument(File file) async {
    String fileName = 'cert_docs/${DateTime.now().millisecondsSinceEpoch}.pdf';
    Reference ref = _storage.ref().child(fileName);
    UploadTask uploadTask = ref.putFile(file);
    TaskSnapshot snapshot = await uploadTask;
    return await snapshot.ref.getDownloadURL();
  }
}