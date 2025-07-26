import numpy as np
import pandas as pd
from scipy.signal import chirp, gausspulse
import os
import sys
sys.path.insert(0, os.path.abspath('.')) 
from src.config import CONFIG

class SyntheticDataGenerator:
    def __init__(self, noise_level=0.01, defect_variability=0.1, num_defects=1):
        self.config = CONFIG['data']
        self.sample_rate = self.config['sample_rate']
        self.fundamental_freq = self.config['fundamental_freq']

        self.noise_level = noise_level
        self.defect_variability = defect_variability
        self.num_defects = num_defects

    def generate_healthy_signal(self):
        """Generate signal for healthy (no debonding) panel"""
        t = np.linspace(0, 1e-3, int(self.sample_rate * 1e-3))
        window = np.hanning(len(t))
        excitation = np.sin(2 * np.pi * self.fundamental_freq * t) * window
        return excitation

    def add_debonding_effect(self, signal, zone, size):
        """Simulate debonding effects based on zone and size"""
        t = np.linspace(0, 1e-3, len(signal))

        # Adjust size with defect variability
        variability = size * self.defect_variability
        size += np.random.uniform(-variability, variability)

        # Add nonlinear harmonics
        if zone in [1, 2, 3]:
            harm_amp = size / 500
            signal += 0.2 * harm_amp * np.sin(2 * np.pi * 2 * self.fundamental_freq * t)
            signal += 0.1 * harm_amp * np.sin(2 * np.pi * 3 * self.fundamental_freq * t)
        else:
            harm_amp = size / 300
            signal += 0.3 * harm_amp * np.sin(2 * np.pi * 2 * self.fundamental_freq * t)
            signal += 0.2 * harm_amp * np.sin(2 * np.pi * 3 * self.fundamental_freq * t)

        # Zone-based amplitude modulation
        if zone == 1:
            signal *= 0.9 + 0.1 * np.sin(2 * np.pi * 50e3 * t)
        elif zone == 2:
            signal *= 0.8 + 0.2 * np.sin(2 * np.pi * 45e3 * t)
        elif zone == 3:
            signal *= 0.85 + 0.15 * np.sin(2 * np.pi * 60e3 * t)

        return signal

    def generate_sample(self, zone, size):
        """Generate one sample with multiple defects if configured"""
        signal = self.generate_healthy_signal()

        for _ in range(self.num_defects if zone > 0 else 1):
            if zone > 0:
                signal = self.add_debonding_effect(signal, zone, size)

        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_level, size=len(signal))
        signal += noise

        return signal

    def generate_dataset(self, num_samples_per_class=50):
        """Generate complete synthetic dataset"""
        signals, zones, sizes = [], [], []

        # Healthy
        for _ in range(num_samples_per_class):
            signals.append(self.generate_healthy_signal())
            zones.append(0)
            sizes.append(0)

        # Debonded
        zone_params = {
            1: {'min_size': 64, 'max_size': 256},
            2: {'min_size': 80, 'max_size': 240},
            3: {'min_size': 120, 'max_size': 288},
            4: {'min_size': 208, 'max_size': 416}
        }

        for zone, params in zone_params.items():
            for _ in range(num_samples_per_class):
                size = np.random.randint(params['min_size'], params['max_size'])
                signals.append(self.generate_sample(zone, size))
                zones.append(zone)
                sizes.append(size)

        return np.array(signals), np.array(zones), np.array(sizes)

    def save_dataset(self, path, signals, zones, sizes):
        """Save dataset to files"""
        if not os.path.exists(path):
            os.makedirs(path)

        for i, signal in enumerate(signals):
            np.save(os.path.join(path, f'signal_{i}.npy'), signal)

        df = pd.DataFrame({'id': [f'signal_{i}' for i in range(len(signals))],
                           'zone': zones,
                           'size': sizes})
        df.to_csv(os.path.join(path, 'labels.csv'), index=False)
