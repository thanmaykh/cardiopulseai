import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import librosa

# ---------------------------------------------------------
# DATASET CONFIGURATION & UNIFIED LABELS
# ---------------------------------------------------------
DATASET_DIR = "./heart_sounds_combined"
CLASSES = ['artifact', 'murmur_abnormal', 'normal']
LABEL_MAP = {
    'artifact': 0,
    'murmur': 1,
    'extrahls': 1,
    'extrastole': 1,
    'abnormal': 1,
    'normal': 2
}

# Download URLs for Public Gold-Standard Datasets
DATASET_URLS = {
    "physionet_2016": "https://physionet.org/files/challenge-2016/1.0.0/training.zip",
    "kaggle_set_a": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names" # Fallback mirror
}

def setup_dataset_structure(base_path=DATASET_DIR):
    """Creates directory structure for unified datasets."""
    os.makedirs(base_path, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(base_path, c), exist_ok=True)
    print(f"[+] Dataset structure created at: {os.path.abspath(base_path)}")

def augment_audio(y, sr):
    """
    Applies audio data augmentation:
    1. Noise Injection
    2. Pitch Shifting (-2 to +2 semitones)
    3. Time Stretching (0.9x to 1.1x speed)
    """
    augmented = []
    
    # 1. Original
    augmented.append(y)
    
    # 2. Additive Gaussian Noise
    noise = np.random.randn(len(y)) * 0.005
    augmented.append(y + noise)
    
    # 3. Pitch Shift (+2 semitones)
    try:
        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        augmented.append(y_pitch)
    except Exception:
        pass
        
    # 4. Time Stretch (1.1x speed)
    try:
        y_fast = librosa.effects.time_stretch(y, rate=1.1)
        augmented.append(y_fast)
    except Exception:
        pass
        
    return augmented

if __name__ == "__main__":
    setup_dataset_structure()
