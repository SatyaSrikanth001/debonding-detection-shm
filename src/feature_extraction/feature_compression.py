import numpy as np
from sklearn.decomposition import PCA
from ..models.autoencoder import build_dcae
from src.config import CONFIG


class FeatureCompressor:
    def __init__(self, method='autoencoder'):
        self.method = method
        if method == 'autoencoder':
            self.autoencoder, self.encoder = build_dcae()
        else:
            self.pca = PCA(n_components=4)
        
    def fit(self, features):
        """Train the compression model"""
        if self.method == 'autoencoder':
            # Reshape for autoencoder (assuming time series data)
            features_reshaped = features.reshape((-1, *features.shape[1:]))
            self.autoencoder.fit(
                features_reshaped, 
                features_reshaped,
                epochs=CONFIG['autoencoder']['epochs'],
                batch_size=CONFIG['autoencoder']['batch_size'],
                verbose=0
            )
        else:
            self.pca.fit(features)
            
    def transform(self, features):
        """Compress features"""
        if self.method == 'autoencoder':
            features_reshaped = features.reshape((-1, *features.shape[1:]))
            return self.encoder.predict(features_reshaped)
        else:
            return self.pca.transform(features)