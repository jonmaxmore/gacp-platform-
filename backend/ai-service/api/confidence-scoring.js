class ConfidenceScoring {
    constructor() {
        this.thresholds = {
            'disease-detection': { high: 0.9, medium: 0.7, low: 0.5 },
            'quality-assessment': { high: 0.85, medium: 0.65, low: 0.4 },
            'yield-prediction': { high: 0.8, medium: 0.6, low: 0.3 },
            'market-optimization': { high: 0.75, medium: 0.55, low: 0.3 }
        };
    }

    calculate(prediction, modelType = 'disease-detection') {
        // Multi-faceted confidence calculation
        const entropy = this.calculateEntropy(prediction);
        const maxProbability = Math.max(...prediction);
        const probabilitySpread = this.calculateSpread(prediction);
        
        // Weighted confidence score
        const baseConfidence = maxProbability;
        const entropyPenalty = entropy / Math.log(prediction.length);
        const spreadBonus = 1 - probabilitySpread;
        
        const confidence = (
            baseConfidence * 0.5 + 
            (1 - entropyPenalty) * 0.3 + 
            spreadBonus * 0.2
        );

        return {
            score: Math.round(confidence * 100) / 100,
            level: this.getConfidenceLevel(confidence, modelType),
            entropy: entropy,
            maxProbability: maxProbability,
            recommendation: this.getRecommendation(confidence, modelType),
            trustworthiness: this.assessTrustworthiness(confidence, prediction)
        };
    }

    calculateEntropy(probabilities) {
        return -probabilities.reduce((entropy, p) => {
            return p > 0 ? entropy + p * Math.log2(p) : entropy;
        }, 0);
    }

    calculateSpread(probabilities) {
        const sorted = [...probabilities].sort((a, b) => b - a);
        return sorted[0] - sorted[1]; // Difference between top 2 predictions
    }

    getConfidenceLevel(confidence, modelType) {
        const thresholds = this.thresholds[modelType];
        if (confidence >= thresholds.high) return 'HIGH';
        if (confidence >= thresholds.medium) return 'MEDIUM';
        if (confidence >= thresholds.low) return 'LOW';
        return 'VERY_LOW';
    }

    getRecommendation(confidence, modelType) {
        const level = this.getConfidenceLevel(confidence, modelType);
        
        const recommendations = {
            'HIGH': 'Proceed with confidence. Result is highly reliable.',
            'MEDIUM': 'Good confidence. Consider additional validation if critical.',
            'LOW': 'Low confidence. Recommend expert consultation or additional testing.',
            'VERY_LOW': 'Very low confidence. Manual verification strongly recommended.'
        };

        return recommendations[level];
    }

    assessTrustworthiness(confidence, prediction) {
        // Advanced trustworthiness assessment
        const variance = this.calculateVariance(prediction);
        const outlierDetection = this.detectOutliers(prediction);
        
        return {
            variance: variance,
            hasOutliers: outlierDetection.hasOutliers,
            distributionHealth: variance < 0.1 ? 'GOOD' : variance < 0.3 ? 'FAIR' : 'POOR',
            overallTrust: confidence > 0.8 && variance < 0.2 ? 'TRUSTWORTHY' : 'QUESTIONABLE'
        };
    }

    calculateVariance(probabilities) {
        const mean = probabilities.reduce((a, b) => a + b, 0) / probabilities.length;
        return probabilities.reduce((variance, p) => variance + Math.pow(p - mean, 2), 0) / probabilities.length;
    }

    detectOutliers(probabilities) {
        const sorted = [...probabilities].sort((a, b) => a - b);
        const q1 = sorted[Math.floor(sorted.length * 0.25)];
        const q3 = sorted[Math.floor(sorted.length * 0.75)];
        const iqr = q3 - q1;
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;
        
        const outliers = probabilities.filter(p => p < lowerBound || p > upperBound);
        
        return {
            hasOutliers: outliers.length > 0,
            outlierCount: outliers.length,
            outlierValues: outliers
        };
    }
}

// /backend/ai-service/monitoring/model-metrics.js
const Redis = require('redis');
const { EventEmitter } = require('events');

class ModelMonitoring extends EventEmitter {
    constructor() {
        super();
        this.redis = Redis.createClient();
        this.metrics = {
            requests: new Map(),
            predictions: new Map(),
            errors: new Map(),
            latency: new Map(),
            accuracy: new Map()
        };
        this.alertThresholds = {
            errorRate: 0.05, // 5% error rate threshold
            avgLatency: 2000, // 2 seconds max average latency
            accuracyDrop: 0.1, // 10% accuracy drop threshold
            confidenceDrop: 0.15 // 15% confidence drop threshold
        };
        this.startTime = Date.now();
        this.setupRedisConnection();
    }

    async setupRedisConnection() {
        try {
            await this.redis.connect();
            console.log('Connected to Redis for monitoring');
        } catch (error) {
            console.error('Redis connection failed:', error);
        }
    }

    logRequest(req, res, duration) {
        const endpoint = req.path;
        const timestamp = Date.now();
        
        // Update request metrics
        if (!this.metrics.requests.has(endpoint)) {
            this.metrics.requests.set(endpoint, {
                count: 0,
                totalDuration: 0,
                errors: 0,
                lastRequest: timestamp
            });
        }

        const requestMetrics = this.metrics.requests.get(endpoint);
        requestMetrics.count++;
        requestMetrics.totalDuration += duration;
        requestMetrics.lastRequest = timestamp;

        if (res.statusCode >= 400) {
            requestMetrics.errors++;
        }

        // Store in Redis for persistence
        this.storeMetrics('request', {
            endpoint,
            duration,
            statusCode: res.statusCode,
            timestamp
        });

        // Check for alerts
        this.checkPerformanceAlerts(endpoint, requestMetrics);
    }

    logPrediction(modelType, result) {
        const timestamp = Date.now();
        
        if (!this.metrics.predictions.has(modelType)) {
            this.metrics.predictions.set(modelType, {
                count: 0,
                totalConfidence: 0,
                avgConfidence: 0,
                confidenceHistory: [],
                lastPrediction: timestamp
            });
        }

        const predictionMetrics = this.metrics.predictions.get(modelType);
        predictionMetrics.count++;
        
        if (result.confidence && typeof result.confidence.score === 'number') {
            predictionMetrics.totalConfidence += result.confidence.score;
            predictionMetrics.avgConfidence = predictionMetrics.totalConfidence / predictionMetrics.count;
            predictionMetrics.confidenceHistory.push({
                score: result.confidence.score,
                timestamp
            });

            // Keep only last 1000 predictions for memory management
            if (predictionMetrics.confidenceHistory.length > 1000) {
                predictionMetrics.confidenceHistory = predictionMetrics.confidenceHistory.slice(-1000);
            }
        }

        predictionMetrics.lastPrediction = timestamp;

        // Store detailed prediction data
        this.storeMetrics('prediction', {
            modelType,
            result,
            timestamp
        });

        // Monitor for model drift
        this.detectModelDrift(modelType, predictionMetrics);
    }

    async storeMetrics(type, data) {
        try {
            const key = `gacp:metrics:${type}:${Date.now()}`;
            await this.redis.setEx(key, 86400, JSON.stringify(data)); // Store for 24 hours
        } catch (error) {
            console.error('Failed to store metrics:', error);
        }
    }

    checkPerformanceAlerts(endpoint, metrics) {
        const errorRate = metrics.errors / metrics.count;
        const avgLatency = metrics.totalDuration / metrics.count;

        if (errorRate > this.alertThresholds.errorRate) {
            this.emit('alert', {
                type: 'HIGH_ERROR_RATE',
                endpoint,
                errorRate,
                threshold: this.alertThresholds.errorRate,
                severity: 'HIGH'
            });
        }

        if (avgLatency > this.alertThresholds.avgLatency) {
            this.emit('alert', {
                type: 'HIGH_LATENCY',
                endpoint,
                avgLatency,
                threshold: this.alertThresholds.avgLatency,
                severity: 'MEDIUM'
            });
        }
    }

    detectModelDrift(modelType, metrics) {
        if (metrics.confidenceHistory.length < 100) return; // Need enough data

        // Calculate recent vs historical confidence trends
        const recent = metrics.confidenceHistory.slice(-50);
        const historical = metrics.confidenceHistory.slice(-200, -50);

        const recentAvg = recent.reduce((sum, item) => sum + item.score, 0) / recent.length;
        const historicalAvg = historical.reduce((sum, item) => sum + item.score, 0) / historical.length;

        const confidenceDrop = historicalAvg - recentAvg;

        if (confidenceDrop > this.alertThresholds.confidenceDrop) {
            this.emit('alert', {
                type: 'MODEL_DRIFT_DETECTED',
                modelType,
                confidenceDrop,
                recentAvg,
                historicalAvg,
                severity: 'HIGH'
            });
        }

        // Statistical drift detection using Kolmogorov-Smirnov test
        const driftScore = this.calculateDriftScore(recent, historical);
        if (driftScore > 0.3) { // Threshold for statistical significance
            this.emit('alert', {
                type: 'STATISTICAL_DRIFT',
                modelType,
                driftScore,
                severity: 'MEDIUM'
            });
        }
    }

    calculateDriftScore(recent, historical) {
        // Simplified drift detection using distribution comparison
        const recentScores = recent.map(item => item.score).sort((a, b) => a - b);
        const historicalScores = historical.map(item => item.score).sort((a, b) => a - b);

        // Calculate empirical cumulative distribution functions
        const maxDifference = Math.max(...recentScores.map((score, i) => {
            const recentCDF = (i + 1) / recentScores.length;
            const historicalCDF = this.calculateCDF(historicalScores, score);
            return Math.abs(recentCDF - historicalCDF);
        }));

        return maxDifference;
    }

    calculateCDF(sortedArray, value) {
        let count = 0;
        for (const item of sortedArray) {
            if (item <= value) count++;
            else break;
        }
        return count / sortedArray.length;
    }

    async getSystemHealth() {
        const now = Date.now();
        const uptime = now - this.startTime;

        const health = {
            status: 'HEALTHY',
            uptime: uptime,
            timestamp: now,
            services: {},
            alerts: []
        };

        // Check each model's health
        for (const [modelType, metrics] of this.metrics.predictions) {
            const timeSinceLastPrediction = now - metrics.lastPrediction;
            const isStale = timeSinceLastPrediction > 300000; // 5 minutes

            health.services[modelType] = {
                status: isStale ? 'STALE' : 'ACTIVE',
                predictions: metrics.count,
                avgConfidence: Math.round(metrics.avgConfidence * 100) / 100,
                lastActivity: metrics.lastPrediction,
                timeSinceLastActivity: timeSinceLastPrediction
            };

            if (isStale) {
                health.status = 'DEGRADED';
                health.alerts.push({
                    type: 'STALE_MODEL',
                    model: modelType,
                    timeSinceLastActivity: timeSinceLastPrediction
                });
            }
        }

        // Check Redis connection
        try {
            await this.redis.ping();
            health.services.redis = { status: 'CONNECTED' };
        } catch (error) {
            health.services.redis = { status: 'DISCONNECTED', error: error.message };
            health.status = 'DEGRADED';
        }

        return health;
    }

    async getMetrics() {
        const metrics = {
            requests: {},
            predictions: {},
            system: {
                uptime: Date.now() - this.startTime,
                timestamp: Date.now()
            }
        };

        // Process request metrics
        for (const [endpoint, data] of this.metrics.requests) {
            metrics.requests[endpoint] = {
                totalRequests: data.count,
                errorCount: data.errors,
                errorRate: Math.round((data.errors / data.count) * 100) / 100,
                avgLatency: Math.round(data.totalDuration / data.count),
                lastRequest: data.lastRequest
            };
        }

        // Process prediction metrics
        for (const [modelType, data] of this.metrics.predictions) {
            const confidenceHistory = data.confidenceHistory.slice(-100); // Last 100 predictions
            
            metrics.predictions[modelType] = {
                totalPredictions: data.count,
                avgConfidence: Math.round(data.avgConfidence * 100) / 100,
                recentConfidenceHigh: confidenceHistory.filter(p => p.score > 0.8).length,
                recentConfidenceMedium: confidenceHistory.filter(p => p.score > 0.6 && p.score <= 0.8).length,
                recentConfidenceLow: confidenceHistory.filter(p => p.score <= 0.6).length,
                lastPrediction: data.lastPrediction,
                confidenceTrend: this.calculateConfidenceTrend(confidenceHistory)
            };
        }

        return metrics;
    }

    calculateConfidenceTrend(confidenceHistory) {
        if (confidenceHistory.length < 10) return 'INSUFFICIENT_DATA';

        const recent = confidenceHistory.slice(-10);
        const older = confidenceHistory.slice(-20, -10);

        if (older.length < 10) return 'INSUFFICIENT_DATA';

        const recentAvg = recent.reduce((sum, item) => sum + item.score, 0) / recent.length;
        const olderAvg = older.reduce((sum, item) => sum + item.score, 0) / older.length;

        const difference = recentAvg - olderAvg;

        if (difference > 0.05) return 'IMPROVING';
        if (difference < -0.05) return 'DECLINING';
        return 'STABLE';
    }

    startPerformanceMonitoring() {
        // Set up alert handling
        this.on('alert', (alert) => {
            console.warn('🚨 AI Model Alert:', alert);
            this.handleAlert(alert);
        });

        // Periodic health checks
        setInterval(async () => {
            const health = await this.getSystemHealth();
            if (health.status !== 'HEALTHY') {
                console.warn('⚠️ System Health Warning:', health);
            }
        }, 60000); // Check every minute

        // Memory cleanup
        setInterval(() => {
            this.cleanupOldMetrics();
        }, 300000); // Cleanup every 5 minutes

        console.log('🔍 Model monitoring started');
    }

    handleAlert(alert) {
        // In production, this would integrate with alerting systems
        // like PagerDuty, Slack, or email notifications
        
        // Store alert in Redis for dashboard display
        this.storeMetrics('alert', alert);

        // Auto-remediation for certain alert types
        switch (alert.type) {
            case 'HIGH_LATENCY':
                console.log('🔄 Attempting auto-remediation for high latency...');
                // Could trigger model reloading, cache warming, etc.
                break;
            
            case 'MODEL_DRIFT_DETECTED':
                console.log('🔄 Model drift detected, flagging for retraining...');
                // Could trigger automated retraining pipeline
                break;
        }
    }

    cleanupOldMetrics() {
        // Clean up old confidence history to prevent memory leaks
        for (const [modelType, data] of this.metrics.predictions) {
            if (data.confidenceHistory.length > 1000) {
                data.confidenceHistory = data.confidenceHistory.slice(-1000);
            }
        }
    }

    async shutdown() {
        try {
            await this.redis.quit();
            console.log('Model monitoring shutdown complete');
        } catch (error) {
            console.error('Error during monitoring shutdown:', error);
        }
    }
}

// /backend/ai-service/models/model-registry.js
class ModelRegistry {
    constructor() {
        this.models = new Map();
        this.loadModelConfigs();
    }

    loadModelConfigs() {
        // Model registry with versioning and metadata
        const modelConfigs = {
            'disease-detection': {
                'cannabis': {
                    versions: [
                        {
                            version: '1.2',
                            path: '/models/disease-detection/cannabis/v1.2/model.json',
                            accuracy: 0.94,
                            f1Score: 0.92,
                            trainingDate: '2024-12-15',
                            isActive: true,
                            performance: {
                                latency: 145, // ms
                                memoryUsage: 256, // MB
                                accuracy: 0.94
                            }
                        },
                        {
                            version: '1.1',
                            path: '/models/disease-detection/cannabis/v1.1/model.json',
                            accuracy: 0.91,
                            f1Score: 0.89,
                            trainingDate: '2024-11-20',
                            isActive: false
                        }
                    ]
                },
                'turmeric': {
                    versions: [
                        {
                            version: '1.0',
                            path: '/models/disease-detection/turmeric/v1.0/model.json',
                            accuracy: 0.89,
                            f1Score: 0.87,
                            trainingDate: '2024-12-10',
                            isActive: true
                        }
                    ]
                }
            },
            'quality-assessment': {
                'cannabis': {
                    versions: [
                        {
                            version: '2.0',
                            path: '/models/quality-assessment/cannabis/v2.0/model.json',
                            accuracy: 0.92,
                            f1Score: 0.90,
                            trainingDate: '2024-12-18',
                            isActive: true
                        }
                    ]
                }
            },
            'yield-prediction': {
                'general': {
                    versions: [
                        {
                            version: '1.5',
                            path: '/models/yield-prediction/general/v1.5/model.json',
                            mse: 0.08,
                            r2Score: 0.86,
                            trainingDate: '2024-12-12',
                            isActive: true
                        }
                    ]
                }
            }
        };

        this.models = new Map(Object.entries(modelConfigs));
    }

    async getLatestModel(modelType, variant) {
        if (!this.models.has(modelType)) {
            throw new Error(`Model type '${modelType}' not found`);
        }

        const modelGroup = this.models.get(modelType);
        if (!modelGroup[variant]) {
            throw new Error(`Model variant '${variant}' not found for type '${modelType}'`);
        }

        const activeModel = modelGroup[variant].versions.find(v => v.isActive);
        if (!activeModel) {
            throw new Error(`No active model found for '${modelType}/${variant}'`);
        }

        return activeModel;
    }

    async getAllModels() {
        const allModels = {};
        for (const [modelType, variants] of this.models) {
            allModels[modelType] = {};
            for (const [variant, data] of Object.entries(variants)) {
                allModels[modelType][variant] = data.versions;
            }
        }
        return allModels;
    }

    async updateModelStatus(modelType, variant, version, isActive) {
        if (!this.models.has(modelType)) return false;
        
        const modelGroup = this.models.get(modelType);
        if (!modelGroup[variant]) return false;

        const model = modelGroup[variant].versions.find(v => v.version === version);
        if (!model) return false;

        // If activating this model, deactivate others
        if (isActive) {
            modelGroup[variant].versions.forEach(v => v.isActive = v.version === version);
        } else {
            model.isActive = false;
        }

        return true;
    }

    async addModel(modelType, variant, modelData) {
        if (!this.models.has(modelType)) {
            this.models.set(modelType, {});
        }

        const modelGroup = this.models.get(modelType);
        if (!modelGroup[variant]) {
            modelGroup[variant] = { versions: [] };
        }

        modelGroup[variant].versions.push(modelData);
        return true;
    }
}

module.exports = { ConfidenceScoring, ModelMonitoring, ModelRegistry };