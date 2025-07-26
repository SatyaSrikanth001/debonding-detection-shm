import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

class DataValidator:
    def __init__(self, data_path):
        self.data_path = data_path
        self.supported_extensions = ['.mat', '.npy', '.csv', '.pkl']
        
    def detect_file_format(self):
        """Detect the format of uploaded files"""
        files = [f for f in os.listdir(self.data_path) if not f.startswith('.')]
        if not files:
            raise ValueError("No files found in the data directory")
            
        ext = os.path.splitext(files[0])[1].lower()
        if ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {ext}")
        return ext
    
    def load_sample_file(self, ext):
        """Load a sample file based on detected format"""
        files = [f for f in os.listdir(self.data_path) 
                if f.endswith(ext) and not f.startswith('.')]
        sample_file = os.path.join(self.data_path, files[0])
        
        if ext == '.mat':
            return loadmat(sample_file)
        elif ext == '.npy':
            return np.load(sample_file)
        elif ext == '.csv':
            return pd.read_csv(sample_file)
        elif ext == '.pkl':
            return pd.read_pickle(sample_file)
    
    def validate_structure(self, data):
        """Validate the structure of loaded data"""
        if isinstance(data, dict):
            print("MATLAB .mat file structure:")
            for key, value in data.items():
                if not key.startswith('__'):
                    print(f"{key}: {type(value)}")
            return 'mat'
        elif isinstance(data, np.ndarray):
            print(f"NumPy array shape: {data.shape}")
            return 'npy'
        elif isinstance(data, pd.DataFrame):
            print("CSV/DataFrame columns:")
            print(data.columns)
            return 'df'
        else:
            raise ValueError("Unrecognized data structure")
    
    def plot_sample_signal(self, data, data_type):
        """Plot sample signals for verification"""
        plt.figure(figsize=(12, 6))
        
        if data_type == 'mat':
            for i, key in enumerate(list(data.keys())[:3]):
                if not key.startswith('__'):
                    plt.subplot(3, 1, i+1)
                    plt.plot(data[key][:1000])
                    plt.title(f"MATLAB Variable: {key}")
        elif data_type == 'npy':
            plt.plot(data[:1000])
            plt.title("NumPy Array Signal")
        elif data_type == 'df':
            for i, col in enumerate(data.columns[:3]):
                if data[col].dtype in [np.float64, np.int64]:
                    plt.subplot(3, 1, i+1)
                    plt.plot(data[col].values[:1000])
                    plt.title(f"DataFrame Column: {col}")
        
        plt.tight_layout()
        plt.show()
    
    def run_validation(self):
        """Run complete validation pipeline"""
        print(f"Validating data in: {self.data_path}")
        
        ext = self.detect_file_format()
        print(f"Detected file format: {ext}")
        
        data = self.load_sample_file(ext)
        data_type = self.validate_structure(data)
        
        print("\nSample data preview:")
        if data_type == 'df':
            print(data.head())
        else:
            print(data)
        
        self.plot_sample_signal(data, data_type)
        print("\nData validation complete!")

if __name__ == "__main__":
    # Update this path to match your VS Code project structure
    validator = DataValidator(data_path='data/raw')
    validator.run_validation()