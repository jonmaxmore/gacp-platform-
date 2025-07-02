import os
import json
import yaml
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import docker
import kubernetes
from kubernetes import client, config
import mlflow
import redis
from celery import Celery
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import boto3
from google.cloud import storage
import tensorflow as tf

# Database Models
Base = declarative_base()

class ModelRegistry(Base):
    __tablename__ = 'model_registry'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # herb_classifier, disease_detector, quality_assessor
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_date = Column(DateTime, default=datetime.utcnow)
    deployment_status = Column(String(20), default='pending')  # pending, deployed, retired
    model_path = Column(String(500))
    metadata = Column(Text)  # JSON metadata
    is_active = Column(Boolean, default=False)
    
class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prediction_count = Column(Integer, default=0)
    accuracy = Column(Float)
    latency_p95 = Column(Float)
    error_rate = Column(Float)
    drift_score = Column(Float)

class DeploymentHistory(Base):
    __tablename__ = 'deployment_history'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    deployment_type = Column(String(20))  # canary, blue_green, rolling
    deployment_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20))  # success, failed, rollback
    traffic_percentage = Column(Float, default=0.0)
    rollback_reason = Column(Text)

# Configuration Classes
@dataclass
class MLOpsConfig:
    """MLOps system configuration"""
    # Database
    database_url: str = "postgresql://mlops:password@localhost:5432/gacp_mlops"
    
    # Redis for caching and queues
    redis_url: str = "redis://localhost:6379/0"
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    
    # Kubernetes
    k8s_namespace: str = "gacp-mlops"
    k8s_config_path: Optional[str] = None
    
    # Storage
    model_storage_type: str = "local"  # local, s3, gcs
    model_storage_path: str = "/models"
    
    # Monitoring
    prometheus_endpoint: str = "http://localhost:9090"
    grafana_endpoint: str = "http://localhost:3000"
    
    # Deployment
    deployment_strategy: str = "canary"  # canary, blue_green, rolling
    canary_traffic_step: float = 0.1
    rollback_threshold_error_rate: float = 0.05
    rollback_threshold_latency: float = 1000  # ms
    
    # Training
    auto_retrain_threshold_accuracy: float = 0.90
    training_schedule_cron: str = "0 2 * * 0"  # Weekly at 2 AM Sunday

class MLOpsOrchestrator:
    """Main MLOps orchestration system"""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.setup_database()
        self.setup_redis()
        self.setup_kubernetes()
        self.setup_mlflow()
        self.setup_storage()
        self.setup_celery()
        
    def setup_database(self):
        """Initialize database connection"""
        self.engine = create_engine(self.config.database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logging.info("Database initialized")
        
    def setup_redis(self):
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(self.config.redis_url)
        logging.info("Redis initialized")
        
    def setup_kubernetes(self):
        """Initialize Kubernetes client"""
        if self.config.k8s_config_path:
            config.load_kube_config(config_file=self.config.k8s_config_path)
        else:
            config.load_incluster_config()
        
        self.k8s_apps_v1 = client.AppsV1Api()
        self.k8s_core_v1 = client.CoreV1Api()
        self.k8s_networking_v1 = client.NetworkingV1Api()
        logging.info("Kubernetes client initialized")
        
    def setup_mlflow(self):
        """Initialize MLflow tracking"""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        logging.info("MLflow initialized")
        
    def setup_storage(self):
        """Initialize model storage"""
        if self.config.model_storage_type == "s3":
            self.storage_client = boto3.client('s3')
        elif self.config.model_storage_type == "gcs":
            self.storage_client = storage.Client()
        else:
            os.makedirs(self.config.model_storage_path, exist_ok=True)
        logging.info(f"Storage initialized: {self.config.model_storage_type}")
        
    def setup_celery(self):
        """Initialize Celery for background tasks"""
        self.celery_app = Celery(
            'gacp_mlops',
            broker=self.config.redis_url,
            backend=self.config.redis_url
        )
        
        # Configure Celery
        self.celery_app.conf.update(
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
        )

class ModelDeploymentManager:
    """Manage model deployments and traffic routing"""
    
    def __init__(self, orchestrator: MLOpsOrchestrator):
        self.orchestrator = orchestrator
        self.config = orchestrator.config
        
    async def deploy_model(self, model_name: str, model_version: str, 
                          deployment_type: str = None) -> Dict:
        """Deploy model with specified strategy"""
        deployment_type = deployment_type or self.config.deployment_strategy
        
        logging.info(f"Deploying {model_name}:{model_version} using {deployment_type}")
        
        if deployment_type == "canary":
            return await self._canary_deployment(model_name, model_version)
        elif deployment_type == "blue_green":
            return await self._blue_green_deployment(model_name, model_version)
        elif deployment_type == "rolling":
            return await self._rolling_deployment(model_name, model_version)
        else:
            raise ValueError(f"Unknown deployment type: {deployment_type}")
    
    async def _canary_deployment(self, model_name: str, model_version: str) -> Dict:
        """Canary deployment strategy"""
        # Get current active model
        current_model = self._get_active_model(model_name)
        
        # Deploy new version with 0% traffic
        await self._create_k8s_deployment(model_name, model_version, replicas=1)
        
        # Gradually increase traffic
        traffic_steps = [0.05, 0.1, 0.25, 0.5, 1.0]
        
        for traffic_percentage in traffic_steps:
            logging.info(f"Routing {traffic_percentage*100}% traffic to {model_version}")
            
            # Update traffic routing
            await self._update_traffic_routing(model_name, model_version, traffic_percentage)
            
            # Wait and monitor
            await asyncio.sleep(300)  # 5 minutes
            
            # Check metrics
            metrics = await self._get_deployment_metrics(model_name, model_version)
            
            if self._should_rollback(metrics):
                logging.error("Rollback triggered due to poor metrics")
                await self._rollback_deployment(model_name, current_model['version'])
                return {"status": "failed", "reason": "metrics_threshold_exceeded"}
            
        # Complete deployment
        await self._finalize_deployment(model_name, model_version)
        return {"status": "success", "traffic_percentage": 1.0}
    
    async def _blue_green_deployment(self, model_name: str, model_version: str) -> Dict:
        """Blue-green deployment strategy"""
        # Deploy to green environment
        await self._create_k8s_deployment(f"{model_name}-green", model_version, replicas=3)
        
        # Run health checks
        if not await self._health_check(f"{model_name}-green"):
            return {"status": "failed", "reason": "health_check_failed"}
        
        # Switch traffic
        await self._switch_traffic_blue_green(model_name)
        
        # Monitor for a period
        await asyncio.sleep(600)  # 10 minutes
        
        metrics = await self._get_deployment_metrics(model_name, model_version)
        if self._should_rollback(metrics):
            await self._switch_traffic_blue_green(model_name, rollback=True)
            return {"status": "failed", "reason": "metrics_threshold_exceeded"}
        
        # Cleanup old deployment
        await self._cleanup_old_deployment(f"{model_name}-blue")
        return {"status": "success"}
    
    async def _rolling_deployment(self, model_name: str, model_version: str) -> Dict:
        """Rolling deployment strategy"""
        # Update deployment with rolling update
        deployment = self._get_k8s_deployment(model_name)
        deployment.spec.template.spec.containers[0].image = f"gacp/{model_name}:{model_version}"
        
        await self.orchestrator.k8s_apps_v1.patch_namespaced_deployment(
            name=model_name,
            namespace=self.config.k8s_namespace,
            body=deployment
        )
        
        # Wait for rollout
        await self._wait_for_rollout(model_name)
        
        return {"status": "success"}
    
    async def _create_k8s_deployment(self, name: str, version: str, replicas: int = 3):
        """Create Kubernetes deployment"""
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=name),
            spec=client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(
                    match_labels={"app": name}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": name, "version": version}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name=name,
                                image=f"gacp/{name}:{version}",
                                ports=[client.V1ContainerPort(container_port=8080)],
                                resources=client.V1ResourceRequirements(
                                    requests={"cpu": "100m", "memory": "256Mi"},
                                    limits={"cpu": "500m", "memory": "512Mi"}
                                ),
                                liveness_probe=client.V1Probe(
                                    http_get=client.V1HTTPGetAction(
                                        path="/health",
                                        port=8080
                                    ),
                                    initial_delay_seconds=30,
                                    period_seconds=10
                                ),
                                readiness_probe=client.V1Probe(
                                    http_get=client.V1HTTPGetAction(
                                        path="/ready",
                                        port=8080
                                    ),
                                    initial_delay_seconds=5,
                                    period_seconds=5
                                )
                            )
                        ]
                    )
                )
            )
        )
        
        await self.orchestrator.k8s_apps_v1.create_namespaced_deployment(
            namespace=self.config.k8s_namespace,
            body=deployment
        )
    
    def _should_rollback(self, metrics: Dict) -> bool:
        """Determine if deployment should be rolled back"""
        error_rate = metrics.get('error_rate', 0)
        latency_p95 = metrics.get('latency_p95', 0)