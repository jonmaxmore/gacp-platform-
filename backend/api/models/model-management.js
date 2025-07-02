const AWS = require('aws-sdk');
const dynamodb = new AWS.DynamoDB.DocumentClient();
const s3 = new AWS.S3();

class ModelManagementService {
  constructor() {
    this.modelTable = 'gacp-model-performance';
    this.modelBucket = process.env.MODEL_ARTIFACTS_BUCKET;
  }

  async deployModel(modelData) {
    try {
      const modelId = `model_${Date.now()}`;
      const timestamp = Date.now();

      // Upload model artifacts to S3
      const s3Key = `models/${modelId}/model.tflite`;
      await s3.upload({
        Bucket: this.modelBucket,
        Key: s3Key,
        Body: modelData.modelFile,
        ContentType: 'application/octet-stream'
      }).promise();

      // Store model metadata in DynamoDB
      const modelMetadata = {
        model_id: modelId,
        version_timestamp: timestamp,
        accuracy: modelData.accuracy,
        precision: modelData.precision,
        recall: modelData.recall,
        confidence_threshold: modelData.confidenceThreshold,
        s3_location: s3Key,
        status: 'deployed',
        created_at: new Date().toISOString(),
        model_type: modelData.modelType,
        training_data_size: modelData.trainingDataSize
      };

      await dynamodb.put({
        TableName: this.modelTable,
        Item: modelMetadata
      }).promise();

      return {
        success: true,
        modelId: modelId,
        s3Location: s3Key,
        metadata: modelMetadata
      };
    } catch (error) {
      console.error('Model deployment error:', error);
      throw new Error(`Failed to deploy model: ${error.message}`);
    }
  }

  async getModelPerformance(modelId, timeRange = '7d') {
    try {
      const endTime = Date.now();
      const startTime = endTime - this.getTimeRangeMs(timeRange);

      const params = {
        TableName: this.modelTable,
        KeyConditionExpression: 'model_id = :modelId AND version_timestamp BETWEEN :start AND :end',
        ExpressionAttributeValues: {
          ':modelId': modelId,
          ':start': startTime,
          ':end': endTime
        },
        ScanIndexForward: false
      };

      const result = await dynamodb.query(params).promise();
      return this.calculatePerformanceMetrics(result.Items);
    } catch (error) {
      console.error('Error fetching model performance:', error);
      throw error;
    }
  }

  calculatePerformanceMetrics(performanceData) {
    if (!performanceData.length) return null;

    const latest = performanceData[0];
    const historical = performanceData.slice(1);

    return {
      current: {
        accuracy: latest.accuracy,
        precision: latest.precision,
        recall: latest.recall,
        timestamp: latest.version_timestamp
      },
      trend: {
        accuracy_change: this.calculateTrend(historical, 'accuracy'),
        precision_change: this.calculateTrend(historical, 'precision'),
        recall_change: this.calculateTrend(historical, 'recall')
      },
      average: {
        accuracy: this.calculateAverage(performanceData, 'accuracy'),
        precision: this.calculateAverage(performanceData, 'precision'),
        recall: this.calculateAverage(performanceData, 'recall')
      }
    };
  }

  calculateTrend(data, metric) {
    if (data.length < 2) return 0;
    const recent = data[0][metric];
    const previous = data[data.length - 1][metric];
    return ((recent - previous) / previous * 100).toFixed(2);
  }

  calculateAverage(data, metric) {
    const sum = data.reduce((acc, item) => acc + item[metric], 0);
    return (sum / data.length).toFixed(4);
  }

  getTimeRangeMs(range) {
    const ranges = {
      '1d': 24 * 60 * 60 * 1000,
      '7d': 7 * 24 * 60 * 60 * 1000,
      '30d': 30 * 24 * 60 * 60 * 1000
    };
    return ranges[range] || ranges['7d'];
  }
}

module.exports = ModelManagementService;