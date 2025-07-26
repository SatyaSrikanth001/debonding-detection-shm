import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pickle
import sys
sys.path.insert(0, os.path.abspath('.'))  # Adjust path if needed
from src.config import CONFIG

class DataLoader:
    def __init__(self, data_path):
        self.data_path = os.path.normpath(data_path) 
        self.sample_rate = CONFIG['data']['sample_rate']
        
    def load(self):
        """Main loading method that auto-detects file format"""
        
        # DEBUG: Print the path being accessed
        print(f"[DEBUG] Attempting to read data from: {self.data_path}")
        
        # Step 1: Check if path exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"[ERROR] Path does not exist: {self.data_path}")
        
        # Step 2: List all non-hidden files
        try:
            files = [f for f in os.listdir(self.data_path) if not f.startswith('.')]
        except OSError as e:
            raise OSError(f"[ERROR] Failed to list files in {self.data_path}: {e}")
        
        if not files:
            raise ValueError("[ERROR] No data files found in directory.")
        
        # Step 3: Detect file extension
        ext = os.path.splitext(files[0])[1].lower()
        
        print(f"[DEBUG] Found files: {files}")
        print(f"[DEBUG] Detected extension: {ext}")
        
        # Step 4: Load based on file extension
        if ext == '.mat':
            return self._load_mat()
        elif ext == '.npy':
            return self._load_numpy()
        elif ext == '.csv':
            return self._load_csv()
        elif ext == '.pkl':
            return self._load_pickle()
        else:
            raise ValueError(f"[ERROR] Unsupported file format: {ext}")
   
    def _load_mat(self):
        """Load MATLAB .mat files"""
        signals = []
        labels = {'zone': [], 'size': []}
        
        for file in os.listdir(self.data_path):
            if file.endswith('.mat'):
                data = loadmat(os.path.join(self.data_path, file))
                
                # Extract signal (adjust keys based on your actual data)
                if 'signal' in data:
                    signals.append(data['signal'].flatten())
                elif 'waveform' in data:
                    signals.append(data['waveform'].flatten())
                else:
                    # Try to find the first array variable
                    for key, value in data.items():
                        if isinstance(value, np.ndarray) and not key.startswith('__'):
                            signals.append(value.flatten())
                            break
                
                # Extract labels from filename if possible
                # Example: "signal_zone1_size50.mat"
                try:
                    parts = file.split('_')
                    labels['zone'].append(int(parts[1][4:]))
                    labels['size'].append(int(parts[2][4:-4]))
                except:
                    labels['zone'].append(0)
                    labels['size'].append(0)
                    
        return np.array(signals), labels
    
    def _load_numpy(self):
        """Load NumPy .npy files"""
        signals = []
        labels = {'zone': [], 'size': []}
        
        for file in os.listdir(self.data_path):
            if file.endswith('.npy'):
                signals.append(np.load(os.path.join(self.data_path, file)))
                
                # Extract labels from filename
                try:
                    parts = file.split('_')
                    labels['zone'].append(int(parts[1][4:]))
                    labels['size'].append(int(parts[2][4:-4]))
                except:
                    labels['zone'].append(0)
                    labels['size'].append(0)
                    
        return np.array(signals), labels
    
    def _load_csv(self):
        """Load CSV files"""
        dfs = []
        for file in os.listdir(self.data_path):
            if file.endswith('.csv'):
                dfs.append(pd.read_csv(os.path.join(self.data_path, file)))
        
        if not dfs:
            raise ValueError("No CSV files found")
            
        # Assuming one main CSV with all data
        df = pd.concat(dfs)
        
        # Extract signals and labels
        signal_cols = [col for col in df.columns if 'signal' in col.lower()]
        if not signal_cols:
            signal_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
            
        signals = df[signal_cols].values.T
        labels = {
            'zone': df['zone'].values if 'zone' in df.columns else np.zeros(len(df)),
            'size': df['size'].values if 'size' in df.columns else np.zeros(len(df))
        }
        
        return signals, labels
    
    def _load_pickle(self):
        """Load Python pickle files"""
        with open(os.path.join(self.data_path, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
        return data['signals'], data['labels']