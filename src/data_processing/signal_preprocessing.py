import sys
import numpy as np
from scipy import signal as sg  # ✅ Rename to avoid shadowing
import os

sys.path.insert(0, os.path.abspath('.'))

from src.config import CONFIG

class SignalProcessor:
    def __init__(self):
        self.config = CONFIG['data']
        
    def apply_bandpass(self, sig):
        """Apply bandpass filter around excitation frequency"""
        nyq = 0.5 * self.config['sample_rate']
        low = (self.config['fundamental_freq'] - 50e3) / nyq
        high = (self.config['fundamental_freq'] * 3 + 50e3) / nyq
        b, a = sg.butter(5, [low, high], btype='band')  # ✅ use sg
        return sg.filtfilt(b, a, sig)                   # ✅ use sg
    
    def normalize(self, sig):
        """Normalize signal to [-1, 1] range"""
        return 2 * (sig - np.min(sig)) / (np.max(sig) - np.min(sig)) - 1
    
    def process(self, sig):
        """Complete preprocessing pipeline"""
        sig = self.apply_bandpass(sig)
        sig = self.normalize(sig)
        return sig
