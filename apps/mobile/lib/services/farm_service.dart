class FarmService {
  final FirebaseFirestore _firestore = FirebaseService().firestore;

  // Create a new farm
  Future<void> createFarm({
    required String name,
    required GeoPoint location,
    required String ownerId,
    required List<String> herbs,
  }) async {
    await _firestore.collection('farms').add({
      'name': name,
      'location': location,
      'ownerId': ownerId,
      'herbs': herbs,
      'createdAt': FieldValue.serverTimestamp(),
      'updatedAt': FieldValue.serverTimestamp(),
    });
  }

  // Get farms for a user
  Stream<QuerySnapshot> getUserFarms(String userId) {
    return _firestore
        .collection('farms')
        .where('ownerId', isEqualTo: userId)
        .snapshots();
  }
}