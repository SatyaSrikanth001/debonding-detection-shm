import numpy as np
import joblib
import gc
import os
from datetime import datetime
from tqdm import tqdm
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, 
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_processing.data_augmentation import DataAugmenter
from src.feature_extraction.harmonic_features import HarmonicFeatureExtractor
from src.models.autoencoder import build_dcae
from src.models.stacked_model import build_stacked_model
from src.config import CONFIG

def clear_keras_session():
    """Enhanced memory clearing with optional mixed precision"""
    if CONFIG['training'].get('mixed_precision', False):
        from tensorflow.keras.mixed_precision import set_global_policy
        set_global_policy('mixed_float16')
    
    K.clear_session()
    gc.collect()
    print("🧹 Cleared TensorFlow session and collected garbage.")

class DebondingTrainer:
    def __init__(self):
        self.augmenter = DataAugmenter()
        self.feature_extractor = HarmonicFeatureExtractor()
        
        # Initialize with memory optimization
        clear_keras_session()
        
        # Build models
        self.autoencoder, self.encoder = build_dcae()
        self.model = build_stacked_model()
        
        # Model information
        print("\nAutoencoder Summary:")
        self.autoencoder.summary()
        print("\nClassifier Information:")
        self._print_classifier_info()
    
    def _print_classifier_info(self):
        """Enhanced model information printing"""
        if hasattr(self.model, "summary"):
            self.model.summary()
        elif hasattr(self.model, "estimators"):
            print(f"Ensemble Classifier: {type(self.model).__name__}")
            print(f"Number of base estimators: {len(self.model.estimators)}")
            print("Base estimator types:")
            for i, (name, estimator) in enumerate(self.model.estimators):
                print(f"- Estimator {i+1}: {name} ({type(estimator).__name__})")
        else:
            print(f"Classifier Type: {type(self.model).__name__}")
    
    def process_signals(self, signals, labels):
        """Enhanced signal processing with new augmentations"""
        print("Augmenting signals with new methods...")
        augmented_signals = []
        augmented_labels = []

        # Check if augmenter supports new methods
        augmenter_has_new_methods = hasattr(self.augmenter, 'use_time_warp') and \
                                   hasattr(self.augmenter, 'use_random_noise')

        for signal, label in tqdm(zip(signals, labels), total=len(signals)):
            if augmenter_has_new_methods:
                augmented = self.augmenter.augment_signal(
                    signal,
                    use_time_warp=CONFIG['augmentation'].get('time_warping', False),
                    use_random_noise=CONFIG['augmentation'].get('random_noise', False)
                )
            else:
                # Fallback to original augmentation
                augmented = self.augmenter.augment_signal(signal)

            augmented_signals.extend(augmented)
            augmented_labels.extend([label] * len(augmented))

        print("Extracting enhanced features...")
        features = []
        for signal in tqdm(augmented_signals):
            feat = self.feature_extractor.extract(signal)
            features.append(list(feat.values()))

        return np.array(augmented_signals), np.array(features), np.array(augmented_labels)  
          
    def _get_callbacks(self, model_type='autoencoder'):
        """Enhanced callback system with TensorBoard and checkpointing"""
        cfg = CONFIG[model_type]
        callbacks = []
        
        # Early Stopping
        if cfg.get('early_stopping', False):
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=cfg.get('patience', 10),
                restore_best_weights=True,
                verbose=1
            ))
        
        # Learning Rate Reduction
        if cfg.get('reduce_lr', False):
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=cfg.get('reduce_lr_patience', 5),
                min_lr=cfg.get('min_lr', 1e-6),
                verbose=1
            ))
        
        # Model Checkpoint
        if CONFIG['training'].get('checkpoint', False):
            os.makedirs(CONFIG['paths']['models'], exist_ok=True)
            callbacks.append(ModelCheckpoint(
                filepath=os.path.join(
                    CONFIG['paths']['models'],
                    f'best_{model_type}.h5'
                ),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ))
        
        # TensorBoard
        if CONFIG['training'].get('tensorboard', False):
            log_dir = os.path.join(
                CONFIG['paths']['logs'],
                datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            callbacks.append(TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                profile_batch='500,520'
            ))
        
        return callbacks
    
    def train_autoencoder(self, signals):
        """Enhanced autoencoder training with all improvements"""
        signals = signals.reshape((-1, *CONFIG['autoencoder']['input_shape']))
        print("Enhanced Autoencoder input shape:", signals.shape)
        
        history = self.autoencoder.fit(
            signals, signals,
            epochs=CONFIG['autoencoder']['epochs'],
            batch_size=CONFIG['autoencoder']['batch_size'],
            validation_split=CONFIG['autoencoder']['validation_split'],
            callbacks=self._get_callbacks('autoencoder'),
            verbose=1
        )
        
        print("✅ Enhanced autoencoder training complete.")
        clear_keras_session()
        return history
    
    def train_classifier(self, features, labels, class_weight=None):
        """Enhanced classifier training with evaluation"""
        print("\nTraining enhanced classifier with options:")
        print(f"- Class weights: {class_weight}")
        print(f"- Input shape: {features.shape}")
        
        if len(features.shape) == 3:
            features = features.reshape((features.shape[0], -1))
            print(f"Reshaped features to: {features.shape}")
        
        # Keras model training
        if hasattr(self.model, "fit") and not hasattr(self.model, "estimators"):
            print("Training enhanced neural network...")
            
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels,
                test_size=CONFIG['classifier']['validation_split'],
                stratify=labels,
                random_state=42
            )
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=CONFIG['classifier']['epochs'],
                batch_size=CONFIG['classifier']['batch_size'],
                callbacks=self._get_callbacks('classifier'),
                verbose=1,
                class_weight=class_weight
            )
            
            print("✅ Enhanced neural network training complete.")
            clear_keras_session()
            return history
        
        # Sklearn model training
        else:
            print(f"Training enhanced {type(self.model).__name__}")
            
            try:
                if hasattr(self.model, 'set_params'):
                    self.model.set_params(**{
                        est: {'class_weight': 'balanced'}
                        for est in self.model.estimators
                    })
            except Exception as e:
                print(f"Couldn't set class weights: {str(e)}")
            
            self.model.fit(features, labels)
            print(f"✅ Enhanced {type(self.model).__name__} training complete.")
            
            # Enhanced evaluation
            if CONFIG['evaluation']['classification_report']:
                preds = self.model.predict(features)
                print("\nClassification Report:")
                print(classification_report(labels, preds))
            
            if CONFIG['evaluation']['confusion_matrix']:
                self._plot_confusion_matrix(labels, self.model.predict(features))
            
            return None
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Enhanced visualization"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Enhanced Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/enhanced_confusion_matrix.png')
        plt.close()
        print("✅ Saved enhanced confusion matrix to plots/enhanced_confusion_matrix.png")
    
    def save_models(self, path):
        """Enhanced model saving"""
        os.makedirs(path, exist_ok=True)
        
        # Save autoencoder components
        self.autoencoder.save(f"{path}/enhanced_autoencoder.h5")
        self.encoder.save(f"{path}/enhanced_encoder.h5")
        print(f"✅ Saved enhanced models to {path}/")
    
    def load_models(self, path):
        """Enhanced model loading"""
        from tensorflow.keras.models import load_model
        
        try:
            self.autoencoder = load_model(f"{path}/enhanced_autoencoder.h5")
            self.encoder = load_model(f"{path}/enhanced_encoder.h5")
            
            if os.path.exists(f"{path}/enhanced_classifier.h5"):
                self.model = load_model(f"{path}/enhanced_classifier.h5")
            else:
                self.model = joblib.load(f"{path}/enhanced_classifier.joblib")
            
            print("✅ Enhanced models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading enhanced models: {str(e)}")
            raise