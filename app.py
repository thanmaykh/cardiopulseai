import os
import time
import io
import base64
import numpy as np
import librosa
import librosa.display
import matplotlib
import traceback
import wave
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

# Import feature extraction functions from training pipeline
from train_heart_sound_model import extract_features, CLASSES, SAMPLE_RATE, DURATION

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_combined_heart_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Trained Model
model = None
def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            print(f"[*] Loading model from {MODEL_PATH}")
            model = keras.models.load_model(MODEL_PATH)
        else:
            print("[!] Model file not found!")
    return model

CLASS_DESCRIPTIONS = {
    'artifact': {
        'title': 'Recording Artifact / External Noise',
        'severity': 'low',
        'badge': 'Warning: Low Quality',
        'color': '#f59e0b',
        'description': 'The audio recording contains significant background noise, microphone handling friction, or ambient interference. Re-recording in a quiet environment is recommended.'
    },
    'murmur_abnormal': {
        'title': 'Abnormal Heart Sound / Murmur Detected',
        'severity': 'high',
        'badge': 'Clinical Attention Required',
        'color': '#ef4444',
        'description': 'Acoustic features indicate potential cardiac murmur or extra heart sound turbulence (e.g., S3/S4 gallop or valvular stenosis/regurgitation). Professional clinical evaluation is advised.'
    },
    'normal': {
        'title': 'Normal Cardiac Sound Rhythm',
        'severity': 'normal',
        'badge': 'Healthy Heart Rhythm',
        'color': '#10b981',
        'description': 'Clear S1 (lub) and S2 (dub) acoustic heartbeat pulses detected with normal interval periodicity and no significant murmur turbulence detected.'
    }
}

def generate_spectrogram_base64(audio_data, sr=SAMPLE_RATE):
    """Generates a high-resolution mel-spectrogram image as Base64 string."""
    plt.figure(figsize=(10, 3.5), facecolor='#0f172a')
    ax = plt.axes()
    ax.set_facecolor('#0f172a')
    
    # Compute Mel Spectrogram
    S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap='magma')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.title('Acoustic Mel-Spectrogram Analysis', color='#f8fafc', fontsize=12, pad=10)
    plt.xlabel('Time (s)', color='#94a3b8', fontsize=10)
    plt.ylabel('Frequency (Hz)', color='#94a3b8', fontsize=10)
    ax.tick_params(colors='#94a3b8')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=180, facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    buf.seek(0)
    base64_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return base64_str

def generate_waveform_base64(audio_data, sr=SAMPLE_RATE):
    """Generates an audio amplitude waveform plot as Base64 string."""
    plt.figure(figsize=(10, 2.5), facecolor='#0f172a')
    ax = plt.axes()
    ax.set_facecolor('#0f172a')
    
    time_axis = np.linspace(0, len(audio_data) / sr, len(audio_data))
    ax.plot(time_axis, audio_data, color='#38bdf8', linewidth=0.8, alpha=0.85)
    ax.fill_between(time_axis, audio_data, color='#0284c7', alpha=0.25)
    
    plt.title('Audio Waveform Signal', color='#f8fafc', fontsize=12, pad=10)
    plt.xlabel('Time (seconds)', color='#94a3b8', fontsize=10)
    plt.ylabel('Amplitude', color='#94a3b8', fontsize=10)
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, linestyle='--', alpha=0.15, color='#94a3b8')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=180, facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    buf.seek(0)
    base64_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return base64_str

@app.route('/')
def index():
    return render_template('index.html')

from audio_denoiser import clean_heart_sound_audio

# Store latest ESP32 analysis for live dashboard broadcast
latest_esp32_result = None

@app.route('/api/esp32/upload', methods=['POST'])
def esp32_upload():
    """Wireless IoT Endpoint receiving WAV audio from ESP32 Stethoscope over Wi-Fi."""
    global latest_esp32_result
    get_model()
    try:
        if 'file' in request.files:
            file = request.files['file']
            filename = file.filename or f"esp32_rec_{int(time.time())}.wav"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
        else:
            pcm_bytes = request.get_data()
            if not pcm_bytes:
                return jsonify({'error': 'No audio data received from ESP32'}), 400
            filename = f"esp32_raw_{int(time.time())}.wav"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm_bytes)

        denoise_enabled = request.args.get('denoise', 'true').lower() == 'true'

        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
        
        if denoise_enabled:
            y = clean_heart_sound_audio(y, sr=sr, apply_denoise=True)
            
        features = extract_features(y, sr=sr)
        inp = np.expand_dims(features, axis=0)

        probs = model.predict(inp, verbose=0)[0]
        pred_idx = np.argmax(probs)
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx] * 100)

        result_payload = {
            'status': 'success',
            'filename': filename,
            'prediction': pred_class,
            'confidence': round(confidence, 1),
            'probabilities': {CLASSES[i]: float(probs[i] * 100) for i in range(len(CLASSES))},
            'denoised': denoise_enabled,
            'spectrogram': generate_spectrogram_base64(y, sr=sr),
            'waveform': generate_waveform_base64(y, sr=sr),
            'timestamp': time.strftime("%H:%M:%S")
        }

        latest_esp32_result = result_payload
        return jsonify(result_payload)

    except Exception as e:
        print("ESP32 Processing Error:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/esp32/latest')
def get_latest_esp32():
    """Poll endpoint for web dashboard to auto-fetch incoming ESP32 recordings."""
    global latest_esp32_result
    if latest_esp32_result:
        res = latest_esp32_result
        return jsonify(res)
    return jsonify({'status': 'waiting'})

def load_audio_robustly(filepath):
    """Loads audio safely using librosa or scipy fallback."""
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
        return y, sr
    except Exception as e:
        import scipy.io.wavfile as wavfile
        try:
            sr_file, data = wavfile.read(filepath)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if data.dtype == np.int16:
                y = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                y = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                y = (data.astype(np.float32) - 128.0) / 128.0
            else:
                y = data.astype(np.float32)
            if sr_file != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr_file, target_sr=SAMPLE_RATE)
            if len(y) > int(SAMPLE_RATE * DURATION):
                y = y[:int(SAMPLE_RATE * DURATION)]
            return y, SAMPLE_RATE
        except Exception as e2:
            raise RuntimeError(f"Audio decode error: {e}. Please ensure file is valid 16-bit PCM WAV.")

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No audio file provided', 'success': False}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400

        apply_denoise = request.form.get('denoise', 'false').lower() == 'true'

        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Load audio using robust loader
        y, sr = load_audio_robustly(filepath)

        # Apply Denoising Pipeline if enabled
        if apply_denoise:
            print(f"[*] Applying Bandpass (20-500Hz) & Spectral Denoising to {file.filename}...")
            y = clean_heart_sound_audio(y, sr=sr, apply_denoise=True)

        # Extract features
        features = extract_features(y, sr=sr)
        features_input = np.expand_dims(features, axis=0) # (1, time_steps, 120)

        # Inference
        mdl = get_model()
        if mdl is None:
            return jsonify({'error': 'Model not loaded on server', 'success': False}), 500

        probs = mdl.predict(features_input)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx] * 100)

        # Class breakdown
        prob_breakdown = {CLASSES[i]: float(probs[i] * 100) for i in range(len(CLASSES))}

        # Visualizations
        spectrogram_img = generate_spectrogram_base64(y, sr=sr)
        waveform_img = generate_waveform_base64(y, sr=sr)

        class_info = CLASS_DESCRIPTIONS.get(pred_class, {
            'title': pred_class.capitalize(),
            'severity': 'normal',
            'badge': 'Analyzed',
            'color': '#3b82f6',
            'description': 'Audio analyzed successfully.'
        })

        return jsonify({
            'success': True,
            'filename': file.filename,
            'denoised': apply_denoise,
            'prediction': pred_class,
            'confidence': f"{confidence:.1f}%",
            'confidence_val': confidence,
            'probabilities': prob_breakdown,
            'info': class_info,
            'spectrogram': spectrogram_img,
            'waveform': waveform_img
        })

    except Exception as e:
        err_msg = str(e).strip() or type(e).__name__
        print(f"[!] Exception during prediction:\n{traceback.format_exc()}")
        return jsonify({'error': err_msg, 'success': False}), 500

@app.route('/api/samples')
def get_samples():
    """Provides sample audio file paths for instant live testing."""
    sample_files = []
    base_sample_dir = os.path.join(BASE_DIR, "heart_sounds")
    categories = ['normal', 'murmur', 'artifact']
    
    for cat in categories:
        cat_dir = os.path.join(base_sample_dir, cat)
        if os.path.exists(cat_dir):
            wavs = [f for f in os.listdir(cat_dir) if f.endswith('.wav')]
            if wavs:
                # Select crystal-clear representative sample
                chosen_file = wavs[0]
                for candidate in wavs:
                    if '201108222235' in candidate or candidate.startswith('synth_') or '156' in candidate or '201105040918' in candidate:
                        chosen_file = candidate
                        break
                sample_files.append({
                    'category': cat.capitalize(),
                    'filename': chosen_file,
                    'path': f"/sample_file/{cat}/{chosen_file}"
                })
    return jsonify(sample_files)

@app.route('/sample_file/<category>/<filename>')
def serve_sample(category, filename):
    cat_dir = os.path.join(BASE_DIR, "heart_sounds", category.lower())
    return send_from_directory(cat_dir, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[*] Starting Heart Sound AI Diagnostic Web Server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
