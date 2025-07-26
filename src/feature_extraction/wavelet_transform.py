import numpy as np
import pywt
from src.config import CONFIG

class WaveletTransformer:
    def __init__(self, wavelet='morl', scales=None):
        self.wavelet = wavelet
        self.scales = scales if scales else np.arange(1, 128)
        
    def transform(self, signal):
        """Compute continuous wavelet transform"""
        coeffs, freqs = pywt.cwt(signal, self.scales, self.wavelet)
        return np.abs(coeffs)
    
    def get_time_frequency_features(self, signal):
        """Extract statistical features from time-frequency representation"""
        tfrep = self.transform(signal)
        features = {
            'tf_max': np.max(tfrep),
            'tf_mean': np.mean(tfrep),
            'tf_std': np.std(tfrep),
            'tf_energy': np.sum(tfrep**2)
        }
        return features