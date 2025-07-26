import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from src.data_processing.synthetic_data import SyntheticDataGenerator
from src.models.train import DebondingTrainer
from src.evaluation.metrics import evaluate_model
from src.config import CONFIG
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import pandas as pd
import time

class PipelineTimer:
    """Context manager for timing pipeline stages"""
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"⏱️  Stage completed in {elapsed:.2f} seconds")

def validate_shapes(signals, zones, sizes):
    """Validate input shapes before processing"""
    assert len(signals) == len(zones) == len(sizes), \
        f"Shape mismatch: signals({len(signals)}), zones({len(zones)}), sizes({len(sizes)})"
    print(f"✅ Validated shapes - Signals: {signals.shape}, Zones: {zones.shape}, Sizes: {sizes.shape}")

def analyze_class_distribution(labels):
    """Analyze and visualize class distribution"""
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))

    plt.figure(figsize=(10, 5))
    plt.bar(dist.keys(), dist.values())
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(list(dist.keys()))
    plt.savefig('class_distribution.png')
    plt.close()

    print("📊 Class distribution:", dist)
    return dist

def generate_and_train():
    try:
        print("\n" + "="*50)
        print("Step 1: Generating synthetic dataset...")
        with PipelineTimer():
            generator = SyntheticDataGenerator(noise_level=0.05, defect_variability=0.3, num_defects=3)
            signals, zones, sizes = generator.generate_dataset(num_samples_per_class=50)
            validate_shapes(signals, zones, sizes)
            os.makedirs('data/raw', exist_ok=True)
            os.makedirs('saved_models', exist_ok=True)
            generator.save_dataset('data/raw', signals, zones, sizes)
            print(f"✅ Saved {len(signals)} samples to data/raw/")
            print("\nClass distribution in original data:")
            unique, counts = np.unique(zones, return_counts=True)
            print(dict(zip(unique, counts)))

        print("\n" + "="*50)
        print("Step 2: Initializing trainer...")
        with PipelineTimer():
            trainer = DebondingTrainer()

        print("\n" + "="*50)
        print("Step 3: Processing signals (augmentation + feature extraction)...")
        with PipelineTimer():
            augmented_signals, features, augmented_zones = trainer.process_signals(signals, zones)
            print(f"Augmented signals shape: {augmented_signals.shape}")
            print(f"Feature matrix shape: {features.shape if features is not None else 'N/A'}")
            print(f"Augmented zones shape: {augmented_zones.shape}")
            class_dist = analyze_class_distribution(augmented_zones)
            print("\nClass distribution after augmentation:")
            unique, counts = np.unique(augmented_zones, return_counts=True)
            print(dict(zip(unique, counts)))

        print("\n" + "="*50)
        print("Step 4: Training autoencoder...")
        with PipelineTimer():
            try:
                expected_shape = CONFIG['autoencoder']['input_shape']
                print(f"Expected input shape: {expected_shape}")
                reshaped_signals = augmented_signals.reshape((-1, *expected_shape))
                print(f"Reshaped signals for AE: {reshaped_signals.shape}")
                trainer.train_autoencoder(reshaped_signals)
                print("✅ Autoencoder training complete")

                print("\nVisualizing reconstructions...")
                sample_signals = reshaped_signals[:5]
                reconstructions = trainer.autoencoder.predict(sample_signals)

                plt.figure(figsize=(15, 6))
                for i in range(5):
                    plt.subplot(2, 5, i+1)
                    plt.plot(sample_signals[i].flatten())
                    plt.title(f"Original {i+1}")
                    plt.ylim(-1, 1)
                    plt.subplot(2, 5, i+6)
                    plt.plot(reconstructions[i].flatten())
                    plt.title(f"Reconstructed {i+1}")
                    plt.ylim(-1, 1)

                plt.tight_layout()
                plt.savefig('reconstructions.png')
                plt.close()
                print("Reconstruction visualization saved to reconstructions.png")

            except Exception as e:
                print(f"❌ Autoencoder training failed: {str(e)}")
                raise

        print("\n" + "="*50)
        print("Step 5: Training classifier with class weights and cross-validation...")
        with PipelineTimer():
            try:
                compressed_features = trainer.encoder.predict(reshaped_signals)
                print(f"Compressed features shape: {compressed_features.shape}")

                print("\nCalculating class weights...")
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(augmented_zones), y=augmented_zones)
                class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
                print("⚖️ Class weights:", class_weight_dict)

                history = trainer.train_classifier(
                    compressed_features,
                    augmented_zones,
                    class_weight=class_weight_dict,
                    epochs=100,
                    batch_size=64
                )

                # Modified section to handle both Keras and sklearn models
                if history is not None and hasattr(history, 'history'):
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.plot(history.history['accuracy'], label='Train Accuracy')
                    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
                    plt.title('Model Accuracy')
                    plt.ylabel('Accuracy')
                    plt.xlabel('Epoch')
                    plt.legend()

                    plt.subplot(1, 2, 2)
                    plt.plot(history.history['loss'], label='Train Loss')
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                    plt.title('Model Loss')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.legend()

                    plt.tight_layout()
                    plt.savefig('training_history.png')
                    plt.close()
                    print("Training history plot saved to training_history.png")
                else:
                    print("Training curves not available for this model type")

            except Exception as e:
                print(f"❌ Classifier training failed: {str(e)}")
                raise

    except Exception as e:
        print("\n" + "="*50)
        print(f"❌ Critical error in pipeline: {str(e)}")
        print("Debugging info:")
        print(f"Python version: {sys.version}")
        print(f"NumPy version: {np.__version__}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir()}")
        if 'signals' in locals():
            print(f"Signals shape: {signals.shape if hasattr(signals, 'shape') else 'N/A'}")
        raise

if __name__ == "__main__":
    start_time = time.time()
    print("="*50)
    print("Starting Debonding Detection Pipeline")
    print("="*50)
    generate_and_train()
    print("\n" + "="*50)
    print(f"Pipeline completed successfully in {time.time() - start_time:.2f} seconds!")
    print("="*50)