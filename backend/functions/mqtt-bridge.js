const functions = require('firebase-functions');
const admin = require('firebase-admin');
const mqtt = require('mqtt');

admin.initializeApp();

// Connect to MQTT broker
const client = mqtt.connect('mqtt://test.mosquitto.org');

client.on('connect', () => {
  console.log('Connected to MQTT broker');
  client.subscribe('gacp/farms/+/sensors', (err) => {
    if (!err) {
      console.log('Subscribed to sensor topics');
    }
  });
});

client.on('message', (topic, message) => {
  const farmId = topic.split('/')[2];
  const data = JSON.parse(message.toString());
  
  console.log(`Received data for farm: ${farmId}`);
  
  // Add timestamp
  data.timestamp = admin.firestore.FieldValue.serverTimestamp();
  
  // Save to Firestore
  admin.firestore().collection('farms').doc(farmId).collection('sensorData').add(data)
    .then(() => console.log('Data saved to Firestore'))
    .catch(error => console.error('Error saving data:', error));
});

// HTTP function to keep the MQTT connection alive
exports.mqttBridge = functions.https.onRequest((req, res) => {
  res.send("MQTT bridge is running");
});