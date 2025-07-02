const AWS = require('aws-sdk');
const dynamodb = new AWS.DynamoDB.DocumentClient();

class ABTestingAnalyticsService {
  constructor() {
    this.abTestingTable = 'gacp-ab-testing-results';
  }

  async getABTestingResults(experimentId, timeRange = '30d') {
    try {
      const endTime = Date.now();
      const startTime = endTime - this.getTimeRangeMs(timeRange);

      const params = {
        TableName: this.abTestingTable,
        KeyConditionExpression: 'experiment_id = :expId AND timestamp BETWEEN :start AND :end',
        ExpressionAttributeValues: {
          ':expId': experimentId,
          ':start': startTime,
          ':end': endTime
        }
      };

      const result = await dynamodb.query(params).promise();
      return this.analyzeABTestResults(result.Items);
    } catch (error) {
      console.error('Error fetching AB testing results:', error);
      throw error;
    }
  }

  analyzeABTestResults(data) {
    const variantA = data.filter(item => item.variant === 'A');
    const variantB = data.filter(item => item.variant === 'B');

    const analysisA = this.calculateVariantMetrics(variantA);
    const analysisB = this.calculateVariantMetrics(variantB);

    const significance = this.calculateStatisticalSignificance(analysisA, analysisB);

    return {
      experiment_summary: {
        total_participants: data.length,
        variant_a_participants: variantA.length,
        variant_b_participants: variantB.length,
        duration_days: this.calculateDurationDays(data)
      },
      variant_a: analysisA,
      variant_b: analysisB,
      statistical_significance: significance,
      recommendation: this.getRecommendation(analysisA, analysisB, significance)
    };
  }

  calculateVariantMetrics(variantData) {
    if (!variantData.length) return null;

    const conversions = variantData.filter(item => item.conversion).length;
    const conversionRate = (conversions / variantData.length * 100).toFixed(2);

    const totalRevenue = variantData.reduce((sum, item) => 
      sum + (item.metrics?.revenue || 0), 0);
    const avgRevenue = (totalRevenue / variantData.length).toFixed(2);

    const avgEngagement = variantData.reduce((sum, item) => 
      sum + (item.metrics?.engagement_score || 0), 0) / variantData.length;

    return {
      participants: variantData.length,
      conversions: conversions,
      conversion_rate: parseFloat(conversionRate),
      total_revenue: totalRevenue,
      avg_revenue_per_user: parseFloat(avgRevenue),
      avg_engagement_score: parseFloat(avgEngagement.toFixed(2)),
      confidence_interval: this.calculateConfidenceInterval(variantData)
    };
  }

  calculateStatisticalSignificance(variantA, variantB) {
    if (!variantA || !variantB) return null;

    const pA = variantA.conversion_rate / 100;
    const pB = variantB.conversion_rate / 100;
    const nA = variantA.participants;
    const nB = variantB.participants;

    // Calculate pooled standard error
    const pPooled = (variantA.conversions + variantB.conversions) / (nA + nB);
    const se = Math.sqrt(pPooled * (1 - pPooled) * (1/nA + 1/nB));
    
    // Calculate z-score
    const zScore = Math.abs(pA - pB) / se;
    
    // Calculate p-value (two-tailed test)
    const pValue = 2 * (1 - this.normalCDF(Math.abs(zScore)));
    
    return {
      z_score: parseFloat(zScore.toFixed(4)),
      p_value: parseFloat(pValue.toFixed(4)),
      is_significant: pValue < 0.05,
      confidence_level: parseFloat(((1 - pValue) * 100).toFixed(2))
    };
  }

  normalCDF(x) {
    // Approximation of normal cumulative distribution function
    return 0.5 * (1 + this.erf(x / Math.sqrt(2)));
  }

  erf(x) {
    // Approximation of error function
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;

    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return sign * y;
  }

  calculateConfidenceInterval(data) {
    const conversions = data.filter(item => item.conversion).length;
    const n = data.length;
    const p = conversions / n;
    
    // 95% confidence interval
    const margin = 1.96 * Math.sqrt((p * (1 - p)) / n);
    
    return {
      lower: Math.max(0, parseFloat(((p - margin) * 100).toFixed(2))),
      upper: Math.min(100, parseFloat(((p + margin) * 100).toFixed(2)))
    };
  }

  getRecommendation(variantA, variantB, significance) {
    if (!significance || !significance.is_significant) {
      return {
        decision: 'continue_testing',
        reason: 'No statistically significant difference detected. Continue testing or increase sample size.',
        winner: null
      };
    }

    const winner = variantA.conversion_rate > variantB.conversion_rate ? 'A' : 'B';
    const improvement = Math.abs(variantA.conversion_rate - variantB.conversion_rate);

    return {
      decision: 'implement_winner',
      winner: winner,
      improvement_percentage: parseFloat(improvement.toFixed(2)),
      reason: `Variant ${winner} shows statistically significant improvement of ${improvement.toFixed(2)}% in conversion rate.`,
      confidence: significance.confidence_level
    };
  }

  calculateDurationDays(data) {
    if (!data.length) return 0;
    const timestamps = data.map(item => item.timestamp);
    const minTime = Math.min(...timestamps);
    const maxTime = Math.max(...timestamps);
    return Math.ceil((maxTime - minTime) / (24 * 60 * 60 * 1000));
  }

  getTimeRangeMs(range) {
    const ranges = {
      '7d': 7 * 24 * 60 * 60 * 1000,
      '30d': 30 * 24 * 60 * 60 * 1000,
      '90d': 90 * 24 * 60 * 60 * 1000
    };
    return ranges[range] || ranges['30d'];
  }
}

module.exports = ABTestingAnalyticsService;