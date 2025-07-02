import json
import uuid
import datetime
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import boto3
from scipy import stats
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HerbType(Enum):
    """ประเภทสมุนไพรที่รองรับในระบบ GACP"""
    CANNABIS = "กัญชา"
    TURMERIC = "ขมิ้นชัน" 
    GINGER = "ขิง"
    BLACK_GALINGALE = "กระชายดำ"
    THAI_GINGER = "ไพล"
    KRATOM = "กระท่อม"

class TestType(Enum):
    """ประเภทการทดสอบ"""
    DISEASE_DETECTION = "disease_detection"
    QUALITY_ASSESSMENT = "quality_assessment"
    YIELD_PREDICTION = "yield_prediction"
    CERTIFICATION_SCORING = "certification_scoring"

class TestStatus(Enum):
    """สถานะการทดสอบ"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    DRAFT = "draft"

@dataclass
class ModelVersion:
    """ข้อมูลเวอร์ชันโมเดล"""
    model_id: str
    version: str
    name: str
    description: str
    created_at: datetime.datetime
    model_path: str
    performance_metrics: Dict[str, float]
    herb_types: List[HerbType]

@dataclass
class TestConfiguration:
    """การกำหนดค่าการทดสอบ A/B"""
    test_id: str
    test_name: str
    test_type: TestType
    herb_types: List[HerbType]
    model_a: ModelVersion
    model_b: ModelVersion
    traffic_split: float  # 0.0-1.0 (สัดส่วนที่ไปใช้ Model A)
    start_date: datetime.datetime
    end_date: datetime.datetime
    min_sample_size: int
    significance_level: float
    power: float
    status: TestStatus
    success_metrics: List[str]

@dataclass
class TestResult:
    """ผลการทดสอบ"""
    result_id: str
    test_id: str
    user_id: str
    model_version: str
    herb_type: HerbType
    prediction: Dict[str, Any]
    ground_truth: Optional[Dict[str, Any]]
    confidence_score: float
    response_time_ms: float
    timestamp: datetime.datetime
    metadata: Dict[str, Any]

@dataclass
class TestAnalysis:
    """การวิเคราะห์ผลการทดสอบ"""
    test_id: str
    model_a_metrics: Dict[str, float]
    model_b_metrics: Dict[str, float]
    statistical_significance: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    recommendation: str
    sample_size_a: int
    sample_size_b: int
    analysis_date: datetime.datetime

class ABTestFramework:
    """Framework หลักสำหรับการทำ A/B Testing"""
    
    def __init__(self, aws_region: str = "ap-southeast-1"):
        self.aws_region = aws_region
        self.dynamodb = boto3.resource('dynamodb', region_name=aws_region)
        self.s3 = boto3.client('s3', region_name=aws_region)
        
        # DynamoDB Tables
        self.tests_table = self.dynamodb.Table('gacp-ab-tests')
        self.results_table = self.dynamodb.Table('gacp-ab-results')
        self.models_table = self.dynamodb.Table('gacp-model-registry')
        
        self.active_tests: Dict[str, TestConfiguration] = {}
        self._load_active_tests()

    def _load_active_tests(self):
        """โหลดการทดสอบที่กำลังทำงานอยู่"""
        try:
            response = self.tests_table.scan(
                FilterExpression="test_status = :status",
                ExpressionAttributeValues={":status": TestStatus.ACTIVE.value}
            )
            
            for item in response['Items']:
                test_config = self._parse_test_config(item)
                self.active_tests[test_config.test_id] = test_config
                
            logger.info(f"Loaded {len(self.active_tests)} active tests")
        except Exception as e:
            logger.error(f"Error loading active tests: {e}")

    def create_ab_test(
        self,
        test_name: str,
        test_type: TestType,
        herb_types: List[HerbType],
        model_a_id: str,
        model_b_id: str,
        traffic_split: float = 0.5,
        duration_days: int = 30,
        min_sample_size: int = 1000,
        significance_level: float = 0.05,
        power: float = 0.8
    ) -> str:
        """สร้างการทดสอบ A/B ใหม่"""
        
        test_id = str(uuid.uuid4())
        
        # ดึงข้อมูลโมเดล
        model_a = self._get_model_version(model_a_id)
        model_b = self._get_model_version(model_b_id)
        
        if not model_a or not model_b:
            raise ValueError("ไม่พบโมเดลที่ระบุ")
        
        # สร้าง Test Configuration
        test_config = TestConfiguration(
            test_id=test_id,
            test_name=test_name,
            test_type=test_type,
            herb_types=herb_types,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split,
            start_date=datetime.datetime.now(),
            end_date=datetime.datetime.now() + datetime.timedelta(days=duration_days),
            min_sample_size=min_sample_size,
            significance_level=significance_level,
            power=power,
            status=TestStatus.ACTIVE,
            success_metrics=self._get_default_metrics(test_type)
        )
        
        # บันทึกลง DynamoDB
        self._save_test_config(test_config)
        
        # เพิ่มเข้า active tests
        self.active_tests[test_id] = test_config
        
        logger.info(f"Created A/B test: {test_name} (ID: {test_id})")
        return test_id

    def route_prediction_request(
        self,
        user_id: str,
        herb_type: HerbType,
        test_type: TestType,
        input_data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """กำหนดเส้นทางการทำนายไปยังโมเดลที่เหมาสม"""
        
        # หาการทดสอบที่เหมาะสม
        applicable_test = self._find_applicable_test(herb_type, test_type)
        
        if not applicable_test:
            # ใช้โมเดลปัจจุบัน (production model)
            return self._route_to_production_model(user_id, herb_type, test_type, input_data)
        
        # กำหนดเส้นทางตาม traffic split
        model_version = self._determine_model_assignment(user_id, applicable_test)
        
        # ทำการทำนาย
        prediction_result = self._execute_prediction(
            model_version, herb_type, input_data
        )
        
        # บันทึกผลการทดสอบ
        self._record_test_result(
            applicable_test.test_id,
            user_id,
            model_version,
            herb_type,
            prediction_result,
            input_data
        )
        
        return model_version, prediction_result

    def _find_applicable_test(
        self, 
        herb_type: HerbType, 
        test_type: TestType
    ) -> Optional[TestConfiguration]:
        """หาการทดสอบที่เหมาะสมสำหรับ request นี้"""
        
        for test_config in self.active_tests.values():
            if (test_config.test_type == test_type and 
                herb_type in test_config.herb_types and
                test_config.status == TestStatus.ACTIVE and
                datetime.datetime.now() <= test_config.end_date):
                return test_config
        
        return None

    def _determine_model_assignment(
        self, 
        user_id: str, 
        test_config: TestConfiguration
    ) -> str:
        """กำหนดว่าจะใช้โมเดลไหน (A หรือ B) สำหรับ user นี้"""
        
        # ใช้ hash ของ user_id เพื่อให้ผลลัพธ์สม่ำเสมอ
        hash_value = hash(f"{user_id}_{test_config.test_id}") % 100
        threshold = int(test_config.traffic_split * 100)
        
        if hash_value < threshold:
            return test_config.model_a.model_id
        else:
            return test_config.model_b.model_id

    def _execute_prediction(
        self,
        model_version: str,
        herb_type: HerbType,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """ทำการทำนายด้วยโมเดลที่กำหนด"""
        
        # จำลองการทำนาย (ในระบบจริงจะเรียก API ของโมเดล)
        start_time = datetime.datetime.now()
        
        # จำลองผลการทำนายตามประเภทสมุนไพร
        if herb_type == HerbType.CANNABIS:
            prediction = self._simulate_cannabis_prediction(input_data)
        elif herb_type == HerbType.TURMERIC:
            prediction = self._simulate_turmeric_prediction(input_data)
        elif herb_type == HerbType.GINGER:
            prediction = self._simulate_ginger_prediction(input_data)
        elif herb_type == HerbType.BLACK_GALINGALE:
            prediction = self._simulate_black_galingale_prediction(input_data)
        elif herb_type == HerbType.THAI_GINGER:
            prediction = self._simulate_thai_ginger_prediction(input_data)
        elif herb_type == HerbType.KRATOM:
            prediction = self._simulate_kratom_prediction(input_data)
        else:
            prediction = {"error": "Unsupported herb type"}
        
        end_time = datetime.datetime.now()
        response_time = (end_time - start_time).total_seconds() * 1000
        
        prediction["response_time_ms"] = response_time
        prediction["model_version"] = model_version
        prediction["confidence_score"] = random.uniform(0.7, 0.95)
        
        return prediction

    def _simulate_cannabis_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """จำลองการทำนายสำหรับกัญชา"""
        return {
            "disease_probability": random.uniform(0.1, 0.3),
            "quality_score": random.uniform(7.5, 9.5),
            "thc_content": random.uniform(15.0, 25.0),
            "cbd_content": random.uniform(5.0, 15.0),
            "harvest_readiness": random.choice(["ready", "not_ready", "overripe"]),
            "gacp_compliance": random.uniform(85, 98)
        }

    def _simulate_turmeric_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """จำลองการทำนายสำหรับขมิ้นชัน"""
        return {
            "disease_probability": random.uniform(0.05, 0.25),
            "quality_score": random.uniform(8.0, 9.5),
            "curcumin_content": random.uniform(3.0, 8.0),
            "moisture_level": random.uniform(10, 15),
            "harvest_readiness": random.choice(["ready", "not_ready"]),
            "gacp_compliance": random.uniform(88, 96)
        }

    def _simulate_ginger_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """จำลองการทำนายสำหรับขิง"""
        return {
            "disease_probability": random.uniform(0.08, 0.30),
            "quality_score": random.uniform(7.8, 9.3),
            "gingerol_content": random.uniform(0.5, 2.5),
            "fiber_content": random.uniform(1.0, 3.0),
            "harvest_readiness": random.choice(["ready", "not_ready"]),
            "gacp_compliance": random.uniform(86, 95)
        }

    def _simulate_black_galingale_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """จำลองการทำนายสำหรับกระชายดำ"""
        return {
            "disease_probability": random.uniform(0.06, 0.28),
            "quality_score": random.uniform(8.2, 9.4),
            "active_compounds": random.uniform(2.0, 5.5),
            "oil_content": random.uniform(1.5, 4.0),
            "harvest_readiness": random.choice(["ready", "not_ready"]),
            "gacp_compliance": random.uniform(87, 97)
        }

    def _simulate_thai_ginger_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """จำลองการทำนายสำหรับไพล"""
        return {
            "disease_probability": random.uniform(0.07, 0.26),
            "quality_score": random.uniform(8.0, 9.2),
            "essential_oil": random.uniform(0.8, 2.8),
            "piperine_content": random.uniform(0.3, 1.2),
            "harvest_readiness": random.choice(["ready", "not_ready"]),
            "gacp_compliance": random.uniform(85, 94)
        }

    def _simulate_kratom_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """จำลองการทำนายสำหรับกระท่อม"""
        return {
            "disease_probability": random.uniform(0.09, 0.32),
            "quality_score": random.uniform(7.5, 9.0),
            "alkaloid_content": random.uniform(0.5, 1.8),
            "leaf_maturity": random.choice(["young", "mature", "old"]),
            "harvest_readiness": random.choice(["ready", "not_ready"]),
            "gacp_compliance": random.uniform(82, 93)
        }

    def _record_test_result(
        self,
        test_id: str,
        user_id: str,
        model_version: str,
        herb_type: HerbType,
        prediction_result: Dict[str, Any],
        input_data: Dict[str, Any]
    ):
        """บันทึกผลการทดสอบ"""
        
        result = TestResult(
            result_id=str(uuid.uuid4()),
            test_id=test_id,
            user_id=user_id,
            model_version=model_version,
            herb_type=herb_type,
            prediction=prediction_result,
            ground_truth=None,  # จะอัพเดทภายหลังเมื่อมีข้อมูลจริง
            confidence_score=prediction_result.get("confidence_score", 0.0),
            response_time_ms=prediction_result.get("response_time_ms", 0.0),
            timestamp=datetime.datetime.now(),
            metadata={"input_data": input_data}
        )
        
        # บันทึกลง DynamoDB
        self._save_test_result(result)

    def analyze_test_results(self, test_id: str) -> TestAnalysis:
        """วิเคราะห์ผลการทดสอบ A/B"""
        
        if test_id not in self.active_tests:
            raise ValueError(f"ไม่พบการทดสอบ ID: {test_id}")
        
        test_config = self.active_tests[test_id]
        
        # ดึงข้อมูลผลการทดสอบ
        results_a, results_b = self._get_test_results(test_id)
        
        if len(results_a) < 30 or len(results_b) < 30:
            logger.warning(f"Sample size too small for meaningful analysis (A: {len(results_a)}, B: {len(results_b)})")
        
        # คำนวณ metrics
        metrics_a = self._calculate_metrics(results_a, test_config.test_type)
        metrics_b = self._calculate_metrics(results_b, test_config.test_type)
        
        # การทดสอบทางสถิติ
        statistical_significance = self._calculate_statistical_significance(
            results_a, results_b, test_config.success_metrics
        )
        
        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            results_a, results_b, test_config.success_metrics
        )
        
        # คำแนะนำ
        recommendation = self._generate_recommendation(
            metrics_a, metrics_b, statistical_significance
        )
        
        analysis = TestAnalysis(
            test_id=test_id,
            model_a_metrics=metrics_a,
            model_b_metrics=metrics_b,
            statistical_significance=statistical_significance,
            confidence_intervals=confidence_intervals,
            recommendation=recommendation,
            sample_size_a=len(results_a),
            sample_size_b=len(results_b),
            analysis_date=datetime.datetime.now()
        )
        
        return analysis

    def _calculate_metrics(
        self, 
        results: List[TestResult], 
        test_type: TestType
    ) -> Dict[str, float]:
        """คำนวณ metrics ตามประเภทการทดสอบ"""
        
        if not results:
            return {}
        
        metrics = {}
        
        # Metrics ทั่วไป
        metrics["avg_confidence"] = np.mean([r.confidence_score for r in results])
        metrics["avg_response_time"] = np.mean([r.response_time_ms for r in results])
        metrics["total_requests"] = len(results)
        
        if test_type == TestType.DISEASE_DETECTION:
            # Metrics สำหรับการตรวจจับโรค
            disease_probs = [r.prediction.get("disease_probability", 0) for r in results]
            metrics["avg_disease_probability"] = np.mean(disease_probs)
            metrics["disease_detection_rate"] = len([p for p in disease_probs if p > 0.5]) / len(disease_probs)
            
        elif test_type == TestType.QUALITY_ASSESSMENT:
            # Metrics สำหรับการประเมินคุณภาพ
            quality_scores = [r.prediction.get("quality_score", 0) for r in results]
            metrics["avg_quality_score"] = np.mean(quality_scores)
            metrics["high_quality_rate"] = len([s for s in quality_scores if s > 8.0]) / len(quality_scores)
            
        elif test_type == TestType.CERTIFICATION_SCORING:
            # Metrics สำหรับการให้คะแนนมาตรฐาน
            gacp_scores = [r.prediction.get("gacp_compliance", 0) for r in results]
            metrics["avg_gacp_compliance"] = np.mean(gacp_scores)
            metrics["certification_pass_rate"] = len([s for s in gacp_scores if s > 85]) / len(gacp_scores)
        
        return metrics

    def _calculate_statistical_significance(
        self,
        results_a: List[TestResult],
        results_b: List[TestResult],
        success_metrics: List[str]
    ) -> Dict[str, float]:
        """คำนวณนัยสำคัญทางสถิติ"""
        
        significance = {}
        
        for metric in success_metrics:
            try:
                values_a = self._extract_metric_values(results_a, metric)
                values_b = self._extract_metric_values(results_b, metric)
                
                if len(values_a) > 0 and len(values_b) > 0:
                    # T-test สำหรับการเปรียบเทียบค่าเฉลี่ย
                    t_stat, p_value = stats.ttest_ind(values_a, values_b)
                    significance[f"{metric}_p_value"] = p_value
                    significance[f"{metric}_significant"] = p_value < 0.05
                    
            except Exception as e:
                logger.error(f"Error calculating significance for {metric}: {e}")
                significance[f"{metric}_p_value"] = 1.0
                significance[f"{metric}_significant"] = False
        
        return significance

    def _calculate_confidence_intervals(
        self,
        results_a: List[TestResult],
        results_b: List[TestResult],
        success_metrics: List[str]
    ) -> Dict[str, Tuple[float, float]]:
        """คำนวณช่วงความเชื่อมั่น"""
        
        confidence_intervals = {}
        
        for metric in success_metrics:
            try:
                values_a = self._extract_metric_values(results_a, metric)
                values_b = self._extract_metric_values(results_b, metric)
                
                if len(values_a) > 0:
                    ci_a = stats.t.interval(0.95, len(values_a)-1, 
                                          loc=np.mean(values_a), 
                                          scale=stats.sem(values_a))
                    confidence_intervals[f"{metric}_model_a"] = ci_a
                
                if len(values_b) > 0:
                    ci_b = stats.t.interval(0.95, len(values_b)-1,
                                          loc=np.mean(values_b),
                                          scale=stats.sem(values_b))
                    confidence_intervals[f"{metric}_model_b"] = ci_b
                    
            except Exception as e:
                logger.error(f"Error calculating CI for {metric}: {e}")
        
        return confidence_intervals

    def _generate_recommendation(
        self,
        metrics_a: Dict[str, float],
        metrics_b: Dict[str, float],
        significance: Dict[str, float]
    ) -> str:
        """สร้างคำแนะนำจากผลการวิเคราะห์"""
        
        # ตรวจสอบว่าโมเดลไหนดีกว่า
        model_a_wins = 0
        model_b_wins = 0
        significant_differences = 0
        
        key_metrics = ["avg_confidence", "avg_response_time", "avg_quality_score", "avg_gacp_compliance"]
        
        for metric in key_metrics:
            if metric in metrics_a and metric in metrics_b:
                if metric == "avg_response_time":  # ต่ำกว่าดีกว่า
                    if metrics_a[metric] < metrics_b[metric]:
                        model_a_wins += 1
                    else:
                        model_b_wins += 1
                else:  # สูงกว่าดีกว่า
                    if metrics_a[metric] > metrics_b[metric]:
                        model_a_wins += 1
                    else:
                        model_b_wins += 1
                
                # ตรวจสอบนัยสำคัญ
                if significance.get(f"{metric}_significant", False):
                    significant_differences += 1
        
        if significant_differences == 0:
            return "ไม่พบความแตกต่างที่มีนัยสำคัญระหว่างโมเดลทั้งสอง แนะนำให้เก็บข้อมูลเพิ่มเติม"
        elif model_a_wins > model_b_wins:
            return f"โมเดล A มีประสิทธิภาพดีกว่า โมเดล B ในหลายๆ ด้าน ({model_a_wins} vs {model_b_wins}) แนะนำให้ใช้ โมเดล A"
        elif model_b_wins > model_a_wins:
            return f"โมเดล B มีประสิทธิภาพดีกว่า โมเดล A ในหลายๆ ด้าน ({model_b_wins} vs {model_a_wins}) แนะนำให้ใช้ โมเดล B"
        else:
            return "โมเดลทั้งสองมีประสิทธิภาพใกล้เคียงกัน แนะนำให้พิจารณาปัจจัยอื่นๆ เช่น ความเสถียร และต้นทุน"

    def generate_test_report(self, test_id: str) -> Dict[str, Any]:
        """สร้างรายงานการทดสอบ A/B"""
        
        analysis = self.analyze_test_results(test_id)
        test_config = self.active_tests[test_id]
        
        report = {
            "test_info": {
                "test_id": test_id,
                "test_name": test_config.test_name,
                "test_type": test_config.test_type.value,
                "herb_types": [herb.value for herb in test_config.herb_types],
                "duration_days": (datetime.datetime.now() - test_config.start_date).days,
                "status": test_config.status.value
            },
            "model_comparison": {
                "model_a": {
                    "id": test_config.model_a.model_id,
                    "name": test_config.model_a.name,
                    "metrics": analysis.model_a_metrics
                },
                "model_b": {
                    "id": test_config.model_b.model_id,
                    "name": test_config.model_b.name,
                    "metrics": analysis.model_b_metrics
                }
            },
            "statistical_analysis": {
                "sample_sizes": {
                    "model_a": analysis.sample_size_a,
                    "model_b": analysis.sample_size_b
                },
                "significance_tests": analysis.statistical_significance,
                "confidence_intervals": analysis.confidence_intervals
            },
            "recommendation": analysis.recommendation,
            "generated_at": datetime.datetime.now().isoformat()
        }
        
        return report

    def stop_test(self, test_id: str, reason: str = "Manual stop") -> bool:
        """หยุดการทดสอบ A/B"""
        
        if test_id not in self.active_tests:
            logger.error(f"Test {test_id} not found in active tests")
            return False
        
        # อัพเดทสถานะเป็น completed
        self.active_tests[test_id].status = TestStatus.COMPLETED
        
        # บันทึกการเปลี่ยนแปลง
        self._save_test_config(self.active_tests[test_id])
        
        # ลบออกจาก active tests
        del self.active_tests[test_id]
        
        logger.info(f"Stopped test {test_id}. Reason: {reason}")
        return True

    # Helper methods สำหรับการจัดการข้อมูล
    def _get_model_version(self, model_id: str) -> Optional[ModelVersion]:
        """ดึงข้อมูลโมเดลจาก registry"""
        # จำลองข้อมูลโมเดล
        return ModelVersion(
            model_id=model_id,
            version="1.0",
            name=f"Model {model_id}",
            description=f"GACP AI Model for herb analysis - {model_id}",
            created_at=datetime.datetime.now(),
            model_path=f"s3://gacp-models/{model_id}/model.pkl",
            performance_metrics={
                "accuracy": random.uniform(0.85, 0.95),
                "precision": random.uniform(0.80, 0.92),
                "recall": random.uniform(0.78, 0.90),
                "f1_score": random.uniform(0.82, 0.91)
            },
            herb_types=[HerbType.CANNABIS, HerbType.TURMERIC, HerbType.GINGER, 
                       HerbType.BLACK_GALINGALE, HerbType.THAI_GINGER, HerbType.KRATOM]
        )

    def _get_default_metrics(self, test_type: TestType) -> List[str]:
        """กำหนด metrics เริ่มต้นตามประเภทการทดสอบ"""
        if test_type == TestType.DISEASE_DETECTION:
            return ["avg_confidence", "disease_detection_rate", "avg_response_time"]
        elif test_type == TestType.QUALITY_ASSESSMENT:
            return ["avg_quality_score", "high_quality_rate", "avg_confidence"]
        elif test_type == TestType.YIELD_PREDICTION:
            return ["prediction_accuracy", "avg_confidence", "avg_response_time"]
        elif test_type == TestType.CERTIFICATION_SCORING:
            return ["avg_gacp_compliance", "certification_pass_rate", "avg_confidence"]
        else:
            return ["avg_confidence", "avg_response_time"]

    def _save_test_config(self, test_config: TestConfiguration):
        """บันทึกการกำหนดค่าการทดสอบลง DynamoDB"""
        try:
            item = {
                'test_id': test_config.test_id,
                'test_name': test_config.test_name,
                'test_type': test_config.test_type.value,
                'herb_types': [herb.value for herb in test_config.herb_types],
                'model_a_id': test_config.model_a.model_id,
                'model_b_id': test_config.model_b.model_id,
                'traffic_split': test_config.traffic_split,
                'start_date': test_config.start_date.isoformat(),
                'end_date': test_config.end_date.isoformat(),
                'min_sample_size': test_config.min_sample_size,
                'significance_level': test_config.significance_level,
                'power': test_config.power,
                'test_status': test_config.status.value,
                'success_metrics': test_config.success_metrics,
                'created_at': datetime.datetime.now().isoformat()
            }
            
            self.tests_table.put_item(Item=item)
            logger.info(f"Saved test config: {test_config.test_id}")
            
        except Exception as e:
            logger.error(f"Error saving test config: {e}")

    def _save_test_result(self, result: TestResult):
        """บันทึกผลการทดสอบลง DynamoDB"""
        try:
            item = {
                'result_id': result.result_id,
                'test_id': result.test_id,
                'user_id': result.user_id,
                'model_version': result.model_version,
                'herb_type': result.herb_type.value,
                'prediction': json.dumps(result.prediction, default=str),
                'ground_truth': json.dumps(result.ground_truth, default=str) if result.ground_truth else None,
                'confidence_score': result.confidence_score,
                'response_time_ms': result.response_time_ms,
                'timestamp': result.timestamp.isoformat(),
                'metadata': json.dumps(result.metadata, default=str)
            }
            
            self.results_table.put_item(Item=item)
            
        except Exception as e:
            logger.error(f"Error saving test result: {e}")

    def _get_test_results(self, test_id: str) -> Tuple[List[TestResult], List[TestResult]]:
        """ดึงผลการทดสอบสำหรับโมเดล A และ B"""
        try:
            response = self.results_table.query(
                IndexName='test-id-index',
                KeyConditionExpression='test_id = :test_id',
                ExpressionAttributeValues={':test_id': test_id}
            )
            
            results_a = []
            results_b = []
            test_config = self.active_tests[test_id]
            
            for item in response['Items']:
                result = self._parse_test_result(item)
                
                if result.model_version == test_config.model_a.model_id:
                    results_a.append(result)
                elif result.model_version == test_config.model_b.model_id:
                    results_b.append(result)
            
            return results_a, results_b
            
        except Exception as e:
            logger.error(f"Error getting test results: {e}")
            return [], []

    def _parse_test_config(self, item: Dict) -> TestConfiguration:
        """แปลงข้อมูลจาก DynamoDB เป็น TestConfiguration"""
        model_a = self._get_model_version(item['model_a_id'])
        model_b = self._get_model_version(item['model_b_id'])
        
        return TestConfiguration(
            test_id=item['test_id'],
            test_name=item['test_name'],
            test_type=TestType(item['test_type']),
            herb_types=[HerbType(herb) for herb in item['herb_types']],
            model_a=model_a,
            model_b=model_b,
            traffic_split=float(item['traffic_split']),
            start_date=datetime.datetime.fromisoformat(item['start_date']),
            end_date=datetime.datetime.fromisoformat(item['end_date']),
            min_sample_size=int(item['min_sample_size']),
            significance_level=float(item['significance_level']),
            power=float(item['power']),
            status=TestStatus(item['test_status']),
            success_metrics=item['success_metrics']
        )

    def _parse_test_result(self, item: Dict) -> TestResult:
        """แปลงข้อมูลจาก DynamoDB เป็น TestResult"""
        return TestResult(
            result_id=item['result_id'],
            test_id=item['test_id'],
            user_id=item['user_id'],
            model_version=item['model_version'],
            herb_type=HerbType(item['herb_type']),
            prediction=json.loads(item['prediction']),
            ground_truth=json.loads(item['ground_truth']) if item.get('ground_truth') else None,
            confidence_score=float(item['confidence_score']),
            response_time_ms=float(item['response_time_ms']),
            timestamp=datetime.datetime.fromisoformat(item['timestamp']),
            metadata=json.loads(item['metadata'])
        )

    def _route_to_production_model(
        self, 
        user_id: str, 
        herb_type: HerbType, 
        test_type: TestType, 
        input_data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """กำหนดเส้นทางไปยังโมเดล production"""
        production_model_id = f"production_{test_type.value}_{herb_type.value}"
        prediction_result = self._execute_prediction(production_model_id, herb_type, input_data)
        return production_model_id, prediction_result

    def _extract_metric_values(self, results: List[TestResult], metric: str) -> List[float]:
        """ดึงค่า metric จากผลการทดสอบ"""
        values = []
        
        for result in results:
            if metric == "avg_confidence":
                values.append(result.confidence_score)
            elif metric == "avg_response_time":
                values.append(result.response_time_ms)
            elif metric in result.prediction:
                values.append(float(result.prediction[metric]))
            elif metric == "disease_detection_rate":
                disease_prob = result.prediction.get("disease_probability", 0)
                values.append(1.0 if disease_prob > 0.5 else 0.0)
            elif metric == "high_quality_rate":
                quality_score = result.prediction.get("quality_score", 0)
                values.append(1.0 if quality_score > 8.0 else 0.0)
            elif metric == "certification_pass_rate":
                gacp_score = result.prediction.get("gacp_compliance", 0)
                values.append(1.0 if gacp_score > 85 else 0.0)
        
        return values

    def get_active_tests(self) -> Dict[str, Dict[str, Any]]:
        """ดึงรายการการทดสอบที่กำลังทำงาน"""
        active_tests_info = {}
        
        for test_id, test_config in self.active_tests.items():
            active_tests_info[test_id] = {
                "test_name": test_config.test_name,
                "test_type": test_config.test_type.value,
                "herb_types": [herb.value for herb in test_config.herb_types],
                "model_a": test_config.model_a.name,
                "model_b": test_config.model_b.name,
                "traffic_split": test_config.traffic_split,
                "start_date": test_config.start_date.isoformat(),
                "end_date": test_config.end_date.isoformat(),
                "status": test_config.status.value,
                "days_running": (datetime.datetime.now() - test_config.start_date).days
            }
        
        return active_tests_info

    def get_test_performance_summary(self, test_id: str) -> Dict[str, Any]:
        """สรุปประสิทธิภาพการทดสอบแบบเรียลไทม์"""
        if test_id not in self.active_tests:
            return {"error": "Test not found"}
        
        results_a, results_b = self._get_test_results(test_id)
        test_config = self.active_tests[test_id]
        
        summary = {
            "test_id": test_id,
            "test_name": test_config.test_name,
            "current_status": test_config.status.value,
            "total_samples": {
                "model_a": len(results_a),
                "model_b": len(results_b),
                "total": len(results_a) + len(results_b)
            },
            "progress": {
                "target_sample_size": test_config.min_sample_size,
                "completion_percentage": min(100, (len(results_a) + len(results_b)) / test_config.min_sample_size * 100)
            },
            "early_indicators": {}
        }
        
        # คำนวณ metrics เบื้องต้น
        if len(results_a) > 10 and len(results_b) > 10:
            metrics_a = self._calculate_metrics(results_a, test_config.test_type)
            metrics_b = self._calculate_metrics(results_b, test_config.test_type)
            
            summary["early_indicators"] = {
                "model_a_avg_confidence": metrics_a.get("avg_confidence", 0),
                "model_b_avg_confidence": metrics_b.get("avg_confidence", 0),
                "model_a_avg_response_time": metrics_a.get("avg_response_time", 0),
                "model_b_avg_response_time": metrics_b.get("avg_response_time", 0)
            }
        
        return summary


# ตัวอย่างการใช้งาน A/B Testing Framework
def example_usage():
    """ตัวอย่างการใช้งาน Framework"""
    
    # สร้าง Framework instance
    ab_framework = ABTestFramework(aws_region="ap-southeast-1")
    
    print("🚀 GACP Platform A/B Testing Framework")
    print("=====================================")
    
    # สร้างการทดสอบใหม่
    test_id = ab_framework.create_ab_test(
        test_name="Quality Assessment Model Comparison",
        test_type=TestType.QUALITY_ASSESSMENT,
        herb_types=[HerbType.CANNABIS, HerbType.TURMERIC],
        model_a_id="model_v1.2",
        model_b_id="model_v2.0",
        traffic_split=0.5,
        duration_days=14,
        min_sample_size=500
    )
    
    print(f"✅ Created A/B test: {test_id}")
    
    # จำลองการใช้งาน
    print("\n📊 Simulating user requests...")
    
    for i in range(100):
        user_id = f"farmer_{i % 50}"  # 50 เกษตรกร
        herb_type = random.choice([HerbType.CANNABIS, HerbType.TURMERIC])
        
        # ข้อมูลจำลอง
        input_data = {
            "image_url": f"s3://gacp-images/{herb_type.value}_{i}.jpg",
            "location": "Bangkok",
            "weather_data": {
                "temperature": random.uniform(25, 35),
                "humidity": random.uniform(60, 90)
            }
        }
        
        # ทำการทำนาย
        model_used, prediction = ab_framework.route_prediction_request(
            user_id=user_id,
            herb_type=herb_type,
            test_type=TestType.QUALITY_ASSESSMENT,
            input_data=input_data
        )
    
    print("✅ Simulation completed")
    
    # ดูสรุปประสิทธิภาพ
    print("\n📈 Performance Summary:")
    summary = ab_framework.get_test_performance_summary(test_id)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # วิเคราะห์ผลการทดสอบ
    print("\n🔬 Test Analysis:")
    try:
        analysis = ab_framework.analyze_test_results(test_id)
        print(f"Recommendation: {analysis.recommendation}")
        print(f"Model A samples: {analysis.sample_size_a}")
        print(f"Model B samples: {analysis.sample_size_b}")
    except Exception as e:
        print(f"Analysis not ready yet: {e}")
    
    # สร้างรายงาน
    print("\n📋 Generating Report...")
    report = ab_framework.generate_test_report(test_id)
    
    print("✅ A/B Testing Framework Demo Complete!")
    return report


if __name__ == "__main__":
    # รันตัวอย่างการใช้งาน
    example_report = example_usage()