# 1. Load saved models
from keras.models import load_model
import joblib
import numpy as np

from src.data_processing.synthetic_data import SyntheticDataGenerator
from src.feature_extraction.feature_compression import FeatureCompressor
from src.data_processing.signal_generator import SignalGenerator
from src.config import CONFIG

# === Adjust the path if needed ===
autoencoder = load_model('/content/drive/MyDrive/detect/saved_models/autoencoder.h5')
classifier = joblib.load('/content/drive/MyDrive/detect/saved_models/stacked_model.joblib')

# 2. Generate synthetic test signals (or load real ones if you have)
print("Generating synthetic test signals...")
generator = SignalGenerator(CONFIG)
signals, zones = generator.generate(n_samples=10)  # test signals only

# 3. Feature extraction
print("Extracting features...")
extractor = FeatureCompressor(CONFIG)
features = extractor.transform(signals)

# 4. Use encoder to compress features (match autoencoder input shape)
input_shape = CONFIG['autoencoder']['input_shape']  # e.g., (3750, 1)
features_reshaped = features.reshape((-1, *input_shape))

compressed_features = autoencoder.encoder.predict(features_reshaped)

# 5. Predict using classifier
print("Running predictions...")
compressed_features = compressed_features.reshape((compressed_features.shape[0], -1))  # Flatten
predictions = classifier.predict(compressed_features)

print("✅ Predicted zones:", predictions)
print("🟡 True zones:", zones)
