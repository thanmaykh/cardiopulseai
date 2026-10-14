"""
CardioPulse AI - Lightweight TFLite Edge Inference Engine
Target Hardware: Raspberry Pi Zero 2W / ARM32 / ARM64 / Embedded PC
Model File: heart_sound_model.tflite (0.57 MB)
"""

import os
import sys
import numpy as np
import librosa

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("[!] Error: Neither tflite_runtime nor tensorflow is installed.")
        sys.exit(1)

from audio_denoiser import clean_heart_sound_audio

CLASSES = ['artifact', 'murmur_abnormal', 'normal']
SAMPLE_RATE = 16000
DURATION = 5
N_MFCC = 40

class TFLiteHeartClassifier:
    def __init__(self, model_path="C:/Users/Student/Desktop/THEJAS/heart_sound_model.tflite"):
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), "heart_sound_model.tflite")
            
        print(f"[*] Loading TFLite Model from {model_path}...")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def extract_features(self, audio_data, sr=SAMPLE_RATE):
        target_len = sr * DURATION
        if len(audio_data) < target_len:
            audio_data = librosa.util.fix_length(audio_data, size=target_len)
        else:
            audio_data = audio_data[:target_len]
            
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
            
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=N_MFCC)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        features = np.vstack([mfcc, delta, delta2]).T
        
        # Z-score standardization
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-6
        features = (features - mean) / std
        return np.expand_dims(features.astype(np.float32), axis=0)

    def predict(self, audio_filepath, denoise=True):
        y, sr = librosa.load(audio_filepath, sr=SAMPLE_RATE, duration=DURATION)
        if denoise:
            y = clean_heart_sound_audio(y, sr=sr, apply_denoise=True)
            
        input_data = self.extract_features(y, sr=sr)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        pred_idx = np.argmax(output_data)
        pred_class = CLASSES[pred_idx]
        confidence = float(output_data[pred_idx] * 100)
        
        return {
            'prediction': pred_class,
            'confidence': round(confidence, 1),
            'probabilities': {CLASSES[i]: float(output_data[i] * 100) for i in range(len(CLASSES))}
        }

if __name__ == '__main__':
    classifier = TFLiteHeartClassifier()
    sample_wav = "C:/Users/Student/Desktop/THEJAS/heart_sounds/normal/synth_normal_0001.wav"
    result = classifier.predict(sample_wav)
    print("\n[+] Edge TFLite Prediction Result:", result)
