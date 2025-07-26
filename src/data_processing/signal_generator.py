import numpy as np

class SignalGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self, n_samples=10):
        input_shape = self.config['autoencoder']['input_shape']
        signal_length = input_shape[0]
        
        # Generate random synthetic signals
        signals = np.random.rand(n_samples, signal_length)
        
        # For simplicity, assign random zones (0, 1, 2, ...)
        zones = np.random.randint(0, self.config['model']['n_classes'], size=n_samples)

        return signals, zones
