const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const multer = require('multer');
const sharp = require('sharp');
const { ModelRegistry } = require('../models/model-registry');
const { ConfidenceScoring } = require('./confidence-scoring');
const { ModelMonitoring } = require('../monitoring/model-metrics');

class AIInferenceServer {
    constructor() {
        this.app = express();
        this.modelRegistry = new ModelRegistry();
        this.confidenceScoring = new ConfidenceScoring();
        this.monitoring = new ModelMonitoring();
        this.models = new Map();
        this.setupMiddleware();
        this.setupRoutes();
    }

    setupMiddleware() {
        this.app.use(express.json());
        this.app.use(multer({ dest: 'uploads/' }).single('image'));
        
        // Rate limiting
        const rateLimit = require('express-rate-limit');
        this.app.use('/predict', rateLimit({
            windowMs: 15 * 60 * 1000, // 15 minutes
            max: 100 // limit each IP to 100 requests per windowMs
        }));

        // Request logging
        this.app.use((req, res, next) => {
            const startTime = Date.now();
            res.on('finish', () => {
                this.monitoring.logRequest(req, res, Date.now() - startTime);
            });
            next();
        });
    }

    setupRoutes() {
        // Disease Detection Endpoint
        this.app.post('/predict/disease', async (req, res) => {
            try {
                const { herbType, imageData, farmerId, location } = req.body;
                
                // Load appropriate model
                const model = await this.loadModel('disease-detection', herbType);
                
                // Preprocess image
                const processedImage = await this.preprocessImage(imageData);
                
                // Make prediction
                const prediction = await model.predict(processedImage).data();
                
                // Calculate confidence score
                const confidence = this.confidenceScoring.calculate(prediction);
                
                // Post-process results
                const result = {
                    prediction: this.interpretPrediction(prediction, herbType),
                    confidence: confidence,
                    recommendations: await this.generateRecommendations(prediction, herbType),
                    timestamp: new Date().toISOString(),
                    farmerId: farmerId,
                    location: location
                };

                // Log prediction for monitoring
                this.monitoring.logPrediction('disease-detection', result);

                res.json(result);

            } catch (error) {
                console.error('Disease prediction error:', error);
                res.status(500).json({ 
                    error: 'Prediction failed', 
                    message: error.message 
                });
            }
        });

        // Quality Assessment Endpoint
        this.app.post('/predict/quality', async (req, res) => {
            try {
                const { herbType, imageData, sensorData, farmerId } = req.body;
                
                const model = await this.loadModel('quality-assessment', herbType);
                
                // Combine image and sensor data
                const features = await this.combineFeatures(imageData, sensorData);
                
                const prediction = await model.predict(features).data();
                const confidence = this.confidenceScoring.calculate(prediction);
                
                const result = {
                    qualityScore: this.calculateQualityScore(prediction),
                    gacpCompliance: this.assessGACPCompliance(prediction),
                    confidence: confidence,
                    defects: this.identifyDefects(prediction),
                    certificationReady: confidence > 0.85 && this.calculateQualityScore(prediction) > 0.8,
                    timestamp: new Date().toISOString(),
                    farmerId: farmerId
                };

                this.monitoring.logPrediction('quality-assessment', result);
                res.json(result);

            } catch (error) {
                console.error('Quality assessment error:', error);
                res.status(500).json({ 
                    error: 'Quality assessment failed', 
                    message: error.message 
                });
            }
        });

        // Yield Prediction Endpoint
        this.app.post('/predict/yield', async (req, res) => {
            try {
                const { 
                    herbType, 
                    farmData, 
                    weatherData, 
                    soilData, 
                    historicalYield,
                    farmerId 
                } = req.body;
                
                const model = await this.loadModel('yield-prediction', herbType);
                
                // Feature engineering
                const features = await this.engineerYieldFeatures({
                    farmData,
                    weatherData,
                    soilData,
                    historicalYield
                });
                
                const prediction = await model.predict(features).data();
                const confidence = this.confidenceScoring.calculate(prediction);
                
                const result = {
                    predictedYield: prediction[0],
                    confidence: confidence,
                    factors: this.identifyYieldFactors(features, prediction),
                    recommendations: await this.generateYieldRecommendations(prediction, features),
                    riskAssessment: this.assessYieldRisk(prediction, confidence),
                    timeline: this.generateHarvestTimeline(prediction, farmData),
                    timestamp: new Date().toISOString(),
                    farmerId: farmerId
                };

                this.monitoring.logPrediction('yield-prediction', result);
                res.json(result);

            } catch (error) {
                console.error('Yield prediction error:', error);
                res.status(500).json({ 
                    error: 'Yield prediction failed', 
                    message: error.message 
                });
            }
        });

        // Market Optimization Endpoint
        this.app.post('/optimize/market', async (req, res) => {
            try {
                const { 
                    herbType, 
                    quantity, 
                    qualityGrade, 
                    location, 
                    timeframe,
                    farmerId 
                } = req.body;
                
                const optimizer = await this.loadModel('market-optimization', 'general');
                
                // Get market data
                const marketData = await this.getMarketData(herbType, location);
                
                // Optimize pricing and timing
                const optimization = await this.optimizeMarketStrategy({
                    herbType,
                    quantity,
                    qualityGrade,
                    location,
                    timeframe,
                    marketData
                });

                const result = {
                    optimalPrice: optimization.price,
                    optimalTiming: optimization.timing,
                    expectedProfit: optimization.profit,
                    marketTrends: optimization.trends,
                    competitorAnalysis: optimization.competitors,
                    recommendations: optimization.recommendations,
                    confidence: optimization.confidence,
                    timestamp: new Date().toISOString(),
                    farmerId: farmerId
                };

                this.monitoring.logPrediction('market-optimization', result);
                res.json(result);

            } catch (error) {
                console.error('Market optimization error:', error);
                res.status(500).json({ 
                    error: 'Market optimization failed', 
                    message: error.message 
                });
            }
        });

        // Model Health Check
        this.app.get('/health', async (req, res) => {
            const health = await this.monitoring.getSystemHealth();
            res.json(health);
        });

        // Model Performance Metrics
        this.app.get('/metrics', async (req, res) => {
            const metrics = await this.monitoring.getMetrics();
            res.json(metrics);
        });
    }

    async loadModel(modelType, variant) {
        const modelKey = `${modelType}-${variant}`;
        
        if (!this.models.has(modelKey)) {
            const modelInfo = await this.modelRegistry.getLatestModel(modelType, variant);
            const model = await tf.loadLayersModel(modelInfo.path);
            this.models.set(modelKey, model);
        }
        
        return this.models.get(modelKey);
    }

    async preprocessImage(imageData) {
        // Convert image to tensor
        const buffer = Buffer.from(imageData, 'base64');
        const processedBuffer = await sharp(buffer)
            .resize(224, 224)
            .png()
            .toBuffer();
        
        const tensor = tf.node.decodeImage(processedBuffer, 3)
            .expandDims(0)
            .div(255.0);
        
        return tensor;
    }

    interpretPrediction(prediction, herbType) {
        // Herb-specific disease classification
        const diseaseClasses = {
            'cannabis': ['healthy', 'powdery_mildew', 'bud_rot', 'spider_mites', 'nutrient_deficiency'],
            'turmeric': ['healthy', 'leaf_spot', 'rhizome_rot', 'bacterial_wilt'],
            'ginger': ['healthy', 'bacterial_wilt', 'soft_rot', 'leaf_spot'],
            'black_galangal': ['healthy', 'root_rot', 'leaf_blight'],
            'plai': ['healthy', 'fungal_infection', 'pest_damage'],
            'kratom': ['healthy', 'leaf_spot', 'scale_insects', 'anthracnose']
        };

        const classes = diseaseClasses[herbType] || diseaseClasses['cannabis'];
        const maxIndex = prediction.indexOf(Math.max(...prediction));
        
        return {
            disease: classes[maxIndex],
            probability: prediction[maxIndex],
            allProbabilities: classes.map((disease, index) => ({
                disease,
                probability: prediction[index]
            }))
        };
    }

    async generateRecommendations(prediction, herbType) {
        // AI-powered treatment recommendations
        const maxIndex = prediction.indexOf(Math.max(...prediction));
        
        const recommendations = {
            'cannabis': {
                0: ['Continue current care routine', 'Monitor regularly'],
                1: ['Apply fungicide', 'Improve air circulation', 'Reduce humidity'],
                2: ['Remove affected buds', 'Increase ventilation', 'Apply preventive spray'],
                3: ['Use miticide spray', 'Introduce beneficial insects', 'Quarantine affected plants'],
                4: ['Adjust nutrient solution', 'Check pH levels', 'Supplement specific nutrients']
            },
            // Add recommendations for other herbs...
        };

        return recommendations[herbType]?.[maxIndex] || ['Consult agricultural expert', 'Monitor plant health'];
    }

    calculateQualityScore(prediction) {
        // Weighted quality scoring algorithm
        const weights = [0.3, 0.25, 0.2, 0.15, 0.1]; // Adjust based on importance
        return prediction.reduce((score, prob, index) => 
            score + prob * weights[index] || 0, 0
        );
    }

    assessGACPCompliance(prediction) {
        // GACP standard compliance assessment
        const complianceThreshold = 0.75;
        const qualityScore = this.calculateQualityScore(prediction);
        
        return {
            compliant: qualityScore >= complianceThreshold,
            score: qualityScore,
            requirements: this.getGACPRequirements(qualityScore),
            deficiencies: this.identifyDeficiencies(prediction)
        };
    }

    async engineerYieldFeatures(data) {
        // Advanced feature engineering for yield prediction
        const features = [];
        
        // Farm characteristics
        features.push(data.farmData.area);
        features.push(data.farmData.plantCount);
        features.push(data.farmData.plantAge);
        
        // Weather features
        features.push(data.weatherData.avgTemperature);
        features.push(data.weatherData.rainfall);
        features.push(data.weatherData.humidity);
        features.push(data.weatherData.sunlightHours);
        
        // Soil features
        features.push(data.soilData.ph);
        features.push(data.soilData.nitrogen);
        features.push(data.soilData.phosphorus);
        features.push(data.soilData.potassium);
        features.push(data.soilData.organicMatter);
        
        // Historical features
        if (data.historicalYield.length > 0) {
            features.push(data.historicalYield.reduce((a, b) => a + b, 0) / data.historicalYield.length);
            features.push(Math.max(...data.historicalYield));
            features.push(Math.min(...data.historicalYield));
        } else {
            features.push(0, 0, 0);
        }
        
        return tf.tensor2d([features]);
    }

    async optimizeMarketStrategy(data) {
        // Advanced market optimization using reinforcement learning principles
        const marketFactors = {
            seasonality: this.calculateSeasonalityFactor(data.herbType),
            demand: await this.predictDemand(data.herbType, data.location),
            competition: await this.analyzeCompetition(data.herbType, data.location),
            quality: this.mapQualityToPrice(data.qualityGrade)
        };

        // Multi-objective optimization
        const strategies = this.generateMarketStrategies(data, marketFactors);
        const optimalStrategy = this.selectOptimalStrategy(strategies);

        return {
            price: optimalStrategy.price,
            timing: optimalStrategy.timing,
            profit: optimalStrategy.expectedProfit,
            trends: marketFactors.demand.trends,
            competitors: marketFactors.competition,
            recommendations: optimalStrategy.recommendations,
            confidence: optimalStrategy.confidence
        };
    }

    start(port = 3001) {
        this.app.listen(port, () => {
            console.log(`AI Inference Server running on port ${port}`);
            this.monitoring.startPerformanceMonitoring();
        });
    }
}

// Initialize and start server
const aiServer = new AIInferenceServer();
aiServer.start();

module.exports = AIInferenceServer;