import os
import glob
import warnings
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

import tensorflow as tf
try:
    import tf_keras as keras
    from tf_keras import layers, models, callbacks, optimizers
except ImportError:
    try:
        from tensorflow.keras import layers, models, callbacks, optimizers
    except (ImportError, AttributeError):
        from keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIGURATION & HYPERPARAMETERS
# ---------------------------------------------------------
SAMPLE_RATE = 16000
DURATION = 5          # Fixed audio duration in seconds
N_MFCC = 40           # Number of MFCC coefficients
MAX_LEN = int(np.ceil(SAMPLE_RATE * DURATION / 512)) # ~157 time frames for 5 sec @ 16kHz
CLASSES = ['artifact', 'murmur', 'normal']
NUM_CLASSES = len(CLASSES)
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------
# 1. FIXED FEATURE EXTRACTION (PRESERVING TIME DIMENSION)
# ---------------------------------------------------------
def extract_mfcc_sequence(file_path, duration=DURATION, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Extracts time-series MFCC feature sequence (shape: [time_steps, n_mfcc]).
    Fixes the bug in the original notebook where np.mean() destroyed temporal features.
    """
    try:
        # Load audio with fixed duration & target sample rate
        y, sr = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast')
        
        target_len = sr * duration
        # Fix audio length via padding or truncation
        if len(y) < target_len:
            y = librosa.util.fix_length(y, size=target_len)
        else:
            y = y[:target_len]
            
        # Standardize audio amplitude
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
            
        # Extract MFCC sequence -> shape: (n_mfcc, time_steps)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Compute Delta & Delta-Delta coefficients for dynamic context
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack features: (3 * n_mfcc, time_steps)
        combined_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        # Transpose to (time_steps, features)
        features_seq = combined_features.T
        return features_seq

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ---------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING PIPELINE
# ---------------------------------------------------------
def load_dataset(base_dir):
    """
    Loads Set A and Set B audio files, extracts sequential features,
    and returns properly structured X, y arrays.
    """
    X_data = []
    y_data = []
    
    # Class mapping rules matching problem domain
    # 0: artifact, 1: murmur (includes extrahls & extrastole), 2: normal
    label_map = {
        'artifact': 0,
        'murmur': 1,
        'extrahls': 1,
        'extrastole': 1,
        'normal': 2
    }

    folders = [os.path.join(base_dir, 'set_a'), os.path.join(base_dir, 'set_b')]
    
    for folder in folders:
        if not os.path.exists(folder):
            continue
        
        for file in os.listdir(folder):
            if not file.endswith('.wav'):
                continue
                
            # Skip unlabeled test files for supervised training
            if 'unlabelledtest' in file.lower():
                continue
                
            # Determine label from filename prefix
            file_prefix = file.split('__')[0].lower()
            
            target_class = None
            for key in label_map:
                if file_prefix.startswith(key):
                    target_class = label_map[key]
                    break
                    
            if target_class is None:
                continue
                
            file_path = os.path.join(folder, file)
            feat = extract_mfcc_sequence(file_path)
            
            if feat is not None:
                X_data.append(feat)
                y_data.append(target_class)
                
    X = np.array(X_data)
    y = np.array(y_data)
    
    return X, y


# ---------------------------------------------------------
# 3. OPTIMIZED MODEL ARCHITECTURE (CRNN: 1D-CNN + BiLSTM)
# ---------------------------------------------------------
def build_crnn_model(input_shape, num_classes):
    """
    1D CNN extracts local spectral pattern maps;
    Bidirectional LSTM captures rhythmic temporal patterns over time.
    """
    inputs = layers.Input(shape=input_shape)
    
    # 1D Convolutional blocks for spectral feature extraction
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Bidirectional Recurrent Layer for temporal rhythm patterns
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    
    # Fully connected classifier block
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="HeartSound_CRNN")
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ---------------------------------------------------------
# 4. TRAINING & EVALUATION PIPELINE
# ---------------------------------------------------------
def main():
    base_dir = "./heart_sounds" # Adjust path to your dataset location
    
    print("--- 1. Loading Data & Extracting Features ---")
    # Replace with actual data load if files exist locally
    # X, y = load_dataset(base_dir)
    
    # Dummy verification shapes for illustration if files not local
    print("Example input pipeline active.")
    
    # Setup training callbacks
    cb = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint("best_heart_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    print("\n--- Model Architecture ---")
    # Example input shape: (157 time frames, 120 features [40 MFCC + 40 Delta + 40 Delta2])
    dummy_input_shape = (157, 120)
    model = build_crnn_model(dummy_input_shape, NUM_CLASSES)
    model.summary()


if __name__ == "__main__":
    main()
