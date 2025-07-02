import os
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import optuna
from optuna.integration import TFKerasPruningCallback

# Configuration
@dataclass
class TrainingConfig:
    """Training configuration settings"""
    model_name: str
    model_version: str
    data_path: str
    output_path: str
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    learning_rate: float = 0.001
    target_accuracy: float = 0.95
    enable_hyperparameter_tuning: bool = True
    mlflow_tracking_uri: str = "http://localhost:5000"
    
class GACPDataProcessor:
    """Handle GACP specific data processing"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.herb_classes = [
            'สมุนไพรไทย_ฟ้าทะลายโจร',
            'สมุนไพรไทย_ขมิ้นชัน', 
            'สมุนไพรไทย_ขิง',
            'สมุนไพรไทย_กระชาย',
            'สมุนไพรไทย_ตะไคร้',
            'สมุนไพรไทย_ใบยี่หร่า'
            'สมุนไพรไทย_กัญชา'
            'สมุนไพรไทย_กระชายดำ'
            'สมุนไพรไทย_ไพล'
            'สมุนไพรไทย_กระท่อม'
        ]
        
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess GACP herb data"""
        logging.info("Loading GACP herb classification data...")
        
        # Load image data and labels
        X, y = self._load_herb_images()
        
        # Data augmentation for better generalization
        X_augmented, y_augmented = self._apply_data_augmentation(X, y)
        
        # Normalize pixel values
        X_augmented = X_augmented.astype('float32') / 255.0
        
        logging.info(f"Data loaded: {X_augmented.shape[0]} samples, {len(self.herb_classes)} classes")
        return X_augmented, y_augmented
        
    def _load_herb_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load herb images from directory structure"""
        images = []
        labels = []
        
        data_dir = Path(self.config.data_path)
        
        for class_idx, herb_class in enumerate(self.herb_classes):
            class_dir = data_dir / herb_class
            if not class_dir.exists():
                logging.warning(f"Directory not found: {class_dir}")
                continue
                
            for img_path in class_dir.glob("*.jpg"):
                try:
                    # Load and resize image
                    img = tf.keras.preprocessing.image.load_img(
                        str(img_path), target_size=(224, 224)
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    logging.error(f"Error loading image {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def _apply_data_augmentation(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation for training robustness"""
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2]
        )
        
        # Generate augmented data
        augmented_images = []
        augmented_labels = []
        
        for i in range(len(X)):
            # Original image
            augmented_images.append(X[i])
            augmented_labels.append(y[i])
            
            # Generate 2 augmented versions per image
            img_reshaped = X[i].reshape(1, *X[i].shape)
            aug_iter = datagen.flow(img_reshaped, batch_size=1)
            
            for _ in range(2):
                aug_img = next(aug_iter)[0]
                augmented_images.append(aug_img)
                augmented_labels.append(y[i])
        
        return np.array(augmented_images), np.array(augmented_labels)

class GACPModelBuilder:
    """Build GACP-specific models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.num_classes = 6  # Thai herbs
        
    def build_herb_classifier(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """Build herb classification model"""
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation='softmax', name='herb_classification')
        ])
        
        return model
    
    def build_disease_detector(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """Build disease detection model"""
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(2, activation='softmax', name='disease_detection')  # Healthy/Diseased
        ])
        
        return model
    
    def build_quality_assessor(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """Build quality assessment model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax', name='quality_assessment')  # Grade A/B/C
        ])
        
        return model

class AutomatedTrainingPipeline:
    """Main automated training pipeline"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_processor = GACPDataProcessor(config)
        self.model_builder = GACPModelBuilder(config)
        self.setup_logging()
        self.setup_mlflow()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(f"gacp_{self.config.model_name}")
        
    def run_training(self) -> Dict:
        """Execute complete training pipeline"""
        with mlflow.start_run(run_name=f"{self.config.model_name}_v{self.config.model_version}"):
            
            # Log configuration
            mlflow.log_params({
                "model_name": self.config.model_name,
                "model_version": self.config.model_version,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate
            })
            
            # Load and preprocess data
            X, y = self.data_processor.load_and_preprocess_data()
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.validation_split, 
                stratify=y, random_state=42
            )
            
            # Convert labels to categorical
            y_train_cat = tf.keras.utils.to_categorical(y_train)
            y_val_cat = tf.keras.utils.to_categorical(y_val)
            
            # Build model based on model type
            if self.config.model_name == "herb_classifier":
                model = self.model_builder.build_herb_classifier(X.shape[1:])
            elif self.config.model_name == "disease_detector":
                model = self.model_builder.build_disease_detector(X.shape[1:])
            elif self.config.model_name == "quality_assessor":
                model = self.model_builder.build_quality_assessor(X.shape[1:])
            else:
                raise ValueError(f"Unknown model type: {self.config.model_name}")
            
            # Hyperparameter tuning if enabled
            if self.config.enable_hyperparameter_tuning:
                best_params = self.optimize_hyperparameters(X_train, y_train_cat, X_val, y_val_cat)
                self.config.learning_rate = best_params['learning_rate']
                self.config.batch_size = best_params['batch_size']
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Setup callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-7
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{self.config.output_path}/best_model.h5",
                    monitor='val_accuracy',
                    save_best_only=True
                )
            ]
            
            # Train model
            logging.info("Starting model training...")
            history = model.fit(
                X_train, y_train_cat,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_data=(X_val, y_val_cat),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val, y_val_cat, verbose=0)
            
            # Log metrics
            mlflow.log_metrics({
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_loss": val_loss
            })
            
            # Generate predictions for detailed evaluation
            y_pred = model.predict(X_val)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Classification report
            report = classification_report(y_val, y_pred_classes, output_dict=True)
            
            # Log model
            mlflow.tensorflow.log_model(model, "model")
            
            # Save model metadata
            model_metadata = {
                "model_name": self.config.model_name,
                "model_version": self.config.model_version,
                "accuracy": float(val_accuracy),
                "precision": float(val_precision),
                "recall": float(val_recall),
                "training_date": datetime.now().isoformat(),
                "classification_report": report
            }
            
            with open(f"{self.config.output_path}/model_metadata.json", 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            logging.info(f"Training completed. Validation accuracy: {val_accuracy:.4f}")
            
            return model_metadata
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val) -> Dict:
        """Optimize hyperparameters using Optuna"""
        logging.info("Starting hyperparameter optimization...")
        
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
            
            # Build model with suggested parameters
            if self.config.model_name == "herb_classifier":
                model = self.model_builder.build_herb_classifier(X_train.shape[1:])
            elif self.config.model_name == "disease_detector":
                model = self.model_builder.build_disease_detector(X_train.shape[1:])
            else:
                model = self.model_builder.build_quality_assessor(X_train.shape[1:])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train with early stopping
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=20,  # Reduced for hyperparameter tuning
                validation_data=(X_val, y_val),
                callbacks=[
                    TFKerasPruningCallback(trial, 'val_accuracy'),
                    tf.keras.callbacks.EarlyStopping(patience=5)
                ],
                verbose=0
            )
            
            return max(history.history['val_accuracy'])
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        
        logging.info(f"Best hyperparameters: {study.best_params}")
        return study.best_params

# Example usage and configuration
if __name__ == "__main__":
    # Configuration for herb classifier training
    config = TrainingConfig(
        model_name="herb_classifier",
        model_version="1.0",
        data_path="/data/gacp/herbs/",
        output_path="/models/herb_classifier/v1.0/",
        batch_size=32,
        epochs=100,
        target_accuracy=0.95,
        enable_hyperparameter_tuning=True
    )
    
    # Create output directory
    os.makedirs(config.output_path, exist_ok=True)
    
    # Initialize and run training pipeline
    pipeline = AutomatedTrainingPipeline(config)
    results = pipeline.run_training()
    
    print("Training completed!")
    print(f"Final accuracy: {results['accuracy']:.4f}")
    print(f"Model saved to: {config.output_path}")