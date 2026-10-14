import os
import glob
import warnings
import joblib
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
try:
    import tf_keras as keras
    from tf_keras import layers, models, callbacks, optimizers
    from tf_keras.utils import to_categorical
except ImportError:
    try:
        from tensorflow.keras import layers, models, callbacks, optimizers
        from tensorflow.keras.utils import to_categorical
    except (ImportError, AttributeError):
        from keras import layers, models, callbacks, optimizers
        from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RATE = 16000
DURATION = 5
N_MFCC = 40
N_FEATURES = N_MFCC * 3  # 120 features (MFCC + Delta + Delta-Delta)
TIME_STEPS = int(np.ceil(SAMPLE_RATE * DURATION / 512)) # ~157 time steps

CLASSES = ['artifact', 'murmur_abnormal', 'normal']
NUM_CLASSES = len(CLASSES)
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

def extract_features(audio_data, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Extracts time-series MFCC + Delta + Delta-Delta feature sequence.
    Shape: (time_steps, 120)
    """
    target_len = sr * DURATION
    if len(audio_data) < target_len:
        audio_data = librosa.util.fix_length(audio_data, size=target_len)
    else:
        audio_data = audio_data[:target_len]
        
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
        
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    features = np.vstack([mfcc, delta, delta2]).T
    
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True) + 1e-6
    features = (features - mean) / std
    
    return features

def augment_audio_signal(y, sr=SAMPLE_RATE):
    """
    Generates fast vector-augmented audio signals:
    1. Original audio
    2. Additive White Gaussian Noise
    3. Amplitude scaling (85% gain)
    4. Circular time shift
    """
    augmented_signals = [y]
    
    # 2. Additive Gaussian noise
    noise = np.random.normal(0, 0.005, len(y))
    augmented_signals.append(y + noise)
    
    # 3. Amplitude gain
    augmented_signals.append(y * 0.85)
    
    # 4. Circular shift
    shift_len = int(sr * 0.2)
    augmented_signals.append(np.roll(y, shift_len))
        
    return augmented_signals

def _process_file_task(file_path, target_label, apply_augmentation):
    """Helper worker task for multi-core parallel feature extraction."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if apply_augmentation:
            signals = augment_audio_signal(y, sr=sr)
        else:
            signals = [y]
            
        extracted = []
        for sig in signals:
            feat = extract_features(sig, sr=sr)
            extracted.append((feat, target_label))
        return extracted
    except Exception:
        return []

def load_all_datasets(data_dirs, apply_augmentation=True):
    """
    Parallel multi-dataset loader: scan audio datasets (PhysioNet 2016, PASCAL, Kaggle),
    unifies class labels, applies parallel feature extraction & augmentation.
    """
    file_tasks = []
    
    label_mapping = {
        'artifact': 0,
        'murmur': 1,
        'extrahls': 1,
        'extrastole': 1,
        'abnormal': 1,
        'murmur_abnormal': 1,
        'normal': 2
    }

    print("[*] Scanning audio dataset directories...", flush=True)
    
    for base_dir in data_dirs:
        if not os.path.exists(base_dir):
            continue
            
        for root, _, files in os.walk(base_dir):
            for file in files:
                if not file.endswith('.wav'):
                    continue
                    
                if 'unlabelledtest' in file.lower() or 'unlabel' in root.lower():
                    continue
                    
                parent_folder = os.path.basename(root).lower()
                filename_clean = file.lower()
                
                target_label = None
                for key, mapped_idx in label_mapping.items():
                    if key in parent_folder or key in filename_clean:
                        target_label = mapped_idx
                        break
                        
                if target_label is None:
                    continue
                    
                file_path = os.path.join(root, file)
                file_tasks.append((file_path, target_label))

    print(f"[+] Found {len(file_tasks)} WAV audio files. Extracting acoustic features in parallel...", flush=True)
    
    if len(file_tasks) == 0:
        return np.array([]), np.array([])

    # Parallel extraction across CPU cores (processes backend bypasses Python GIL)
    results_nested = joblib.Parallel(n_jobs=-1, prefer="processes", batch_size=32)(
        joblib.delayed(_process_file_task)(f_path, lbl, apply_augmentation)
        for f_path, lbl in file_tasks
    )

    X_list = []
    y_list = []
    for res_group in results_nested:
        for feat, lbl in res_group:
            X_list.append(feat)
            y_list.append(lbl)

    print(f"[+] Multi-dataset loader completed: Extracted {len(X_list)} feature matrices.", flush=True)
    return np.array(X_list), np.array(y_list)

def build_high_accuracy_crnn(input_shape=(TIME_STEPS, N_FEATURES), num_classes=NUM_CLASSES):
    """
    Deep 1D-CNN + BiLSTM CRNN Architecture for Acoustic Cardiac Sound Classification.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Conv Block 1
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Conv Block 2
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Conv Block 3 (Deeper Feature Representation)
    x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Recurrent Block (BiLSTM)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.35)(x)
    
    # Dense Classifier
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="CardioPulse_Deep_CRNN")
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate(X, y, epochs=30, batch_size=32):
    """
    Splits dataset, computes class weights, trains model, and saves best checkpoints.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=SEED, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1111, random_state=SEED, stratify=y_train
    )
    
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)
    
    class_weights_vals = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights_vals))
    print(f"[*] Balanced Class Weights: {class_weight_dict}")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_high_accuracy_crnn(input_shape=input_shape)
    model.summary()
    
    model_save_path = os.path.join(BASE_DIR, "best_combined_heart_model.h5")
    
    cb = [
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=cb,
        verbose=1
    )
    
    print("\n" + "="*50)
    print("           FINAL TEST EVALUATION REPORT          ")
    print("="*50)
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"[+] Test Accuracy : {test_acc * 100:.2f}%")
    print(f"[+] Test Loss     : {test_loss:.4f}")
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    
    # Save model
    model.save(model_save_path)
    print(f"[+] Saved high-performance model to: {model_save_path}")
    
    return model, history

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   CARDIO-PULSE AI: MULTI-DATASET TRAINING & EVALUATION   ")
    print("="*60)
    
    data_search_dirs = [
        os.path.join(BASE_DIR, "heart_sounds_merged"),
        os.path.join(BASE_DIR, "heart_sounds"),
        os.path.join(BASE_DIR, "physionet_datasets")
    ]
    
    X, y = load_all_datasets(data_search_dirs, apply_augmentation=True)
    
    if len(X) > 0:
        print(f"\n[+] Final Training Matrix shape: X = {X.shape}, y = {y.shape}")
        model, history = train_and_evaluate(X, y, epochs=30, batch_size=32)
        
        # Export TFLite model automatically
        try:
            print("\n[*] Exporting updated TFLite model...")
            from export_tflite import export_and_verify_tflite
            export_and_verify_tflite()
        except Exception as e:
            print(f"[!] Warning exporting TFLite: {e}")
    else:
        print("[!] No audio dataset found to train on.")
