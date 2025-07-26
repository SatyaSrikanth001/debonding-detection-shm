import numpy as np
import os   
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath('.'))  # Adjust path if needed

from scipy import signal
from tqdm import tqdm
from src.config import CONFIG


class DataAugmenter:
    def __init__(self):
        self.config = CONFIG['augmentation']
        # Initialize new augmentation parameters from config
        self.time_warp_sigma = self.config.get('time_warp_sigma', 0.2)
        self.noise_level = self.config.get('noise_level', 0.05)
        
    def add_noise(self, signal, snr_db):
        """Add white noise with specified SNR (in dB)"""
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        return signal + noise
    
    def random_subsample(self, signal, factor):
        """Random subsampling with preserved order"""
        indices = np.random.choice(len(signal), size=len(signal)//factor, replace=False)
        return signal[np.sort(indices)]
    
    def time_segment(self, signal):
        """Segment signal into overlapping windows"""
        segments = []
        segment_length = CONFIG['data']['segment_length']
        num_segments = CONFIG['data']['num_segments']
        
        for i in range(num_segments):
            start = i * (len(signal) - segment_length) // max(1, (num_segments - 1))
            end = start + segment_length
            segments.append(signal[start:end])
        return segments
    
    def time_warp(self, signal, sigma=None):
        """Time warping augmentation using cubic spline interpolation"""
        if sigma is None:
            sigma = self.time_warp_sigma
        t = np.arange(len(signal))
        warp_steps = np.arange(len(signal)) + np.random.normal(0, sigma, len(signal))
        return np.interp(t, np.sort(warp_steps), signal[np.argsort(warp_steps)])
    
    def add_random_noise(self, signal, noise_level=None):
        """Add Gaussian noise with specified level"""
        if noise_level is None:
            noise_level = self.noise_level
        return signal + noise_level * np.random.normal(0, 1, len(signal))
    
    def augment_signal(self, signal, use_time_warp=False, use_random_noise=False):
        """
        Enhanced signal augmentation with all methods
        Maintains backward compatibility with original implementation
        """
        augmented = []
        
        # Original augmentation pipeline
        for snr in self.config['snr_levels']:
            noisy = self.add_noise(signal, snr)
            
            for factor in self.config['subsample_factors']:
                subsampled = self.random_subsample(noisy, factor)
                
                # Time segmentation - maintains original behavior
                segments = self.time_segment(subsampled)
                augmented.extend(segments)
        
        # New augmentations
        if use_time_warp or self.config.get('time_warping', False):
            warped = self.time_warp(signal)
            augmented.append(warped)
            
        if use_random_noise or self.config.get('random_noise', False):
            noisy = self.add_random_noise(signal)
            augmented.append(noisy)
        
        # Ensure we don't exceed the configured number of augmentations
        max_augmentations = self.config.get('num_augmentations', 36)
        if len(augmented) > max_augmentations:
            augmented = augmented[:max_augmentations]
            
        return augmented
    
    def batch_augment(self, signals, labels=None):
        """Batch augmentation for multiple signals"""
        augmented_signals = []
        augmented_labels = []
        
        for i, signal in enumerate(tqdm(signals, desc="Augmenting signals")):
            augmented = self.augment_signal(signal)
            augmented_signals.extend(augmented)
            if labels is not None:
                augmented_labels.extend([labels[i]] * len(augmented))
                
        if labels is not None:
            return np.array(augmented_signals), np.array(augmented_labels)
        return np.array(augmented_signals)