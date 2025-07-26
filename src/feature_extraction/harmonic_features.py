import numpy as np
from scipy import fft
from src.config import CONFIG

class HarmonicFeatureExtractor:
    def __init__(self):
        self.config = CONFIG['data']
        
    def extract(self, signal):
        n = len(signal)
        fft_result = fft.fft(signal)
        fft_mag = np.abs(fft_result[:n//2])
        freqs = fft.fftfreq(n, 1/self.config['sample_rate'])[:n//2]
        
        features = {}
        for harmonic in self.config['harmonics']:
            center_freq = self.config['fundamental_freq'] * harmonic
            bw = self.config['bandwidth']
            
            mask = (freqs >= center_freq - bw/2) & (freqs <= center_freq + bw/2)
            harmonic_fft = fft_mag[mask]
            
            if len(harmonic_fft) > 0:
                features[f'h{harmonic}_max'] = np.max(harmonic_fft)
                features[f'h{harmonic}_mean'] = np.mean(harmonic_fft)
                features[f'h{harmonic}_std'] = np.std(harmonic_fft)
                features[f'h{harmonic}_area'] = np.trapz(harmonic_fft)
            else:
                features.update({f'h{harmonic}_{stat}': 0 for stat in ['max', 'mean', 'std', 'area']})
                
        return features