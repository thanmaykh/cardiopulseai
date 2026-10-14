import os
import numpy as np
import scipy.io.wavfile as wavfile

DATASET_DIR = "C:/Users/Student/Desktop/THEJAS/heart_sounds"
SAMPLE_RATE = 16000
DURATION = 5.0 # seconds

def generate_normal_heart_sound(sr=SAMPLE_RATE, duration=DURATION, bpm=72):
    """Generates realistic S1 (lub) and S2 (dub) normal cardiac sound waveform."""
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)
    
    beat_interval = 60.0 / bpm
    beat_times = np.arange(0.2, duration - 0.4, beat_interval)
    
    for bt in beat_times:
        # Add slight natural jitter
        bt += np.random.uniform(-0.02, 0.02)
        
        # S1 Beat (Mitral/Tricuspid closure ~120 Hz)
        s1_mask = (t >= bt) & (t <= bt + 0.12)
        t_s1 = t[s1_mask] - bt
        signal[s1_mask] += 0.8 * np.exp(-t_s1 * 35) * np.sin(2 * np.pi * 120 * t_s1)
        
        # S2 Beat (Aortic/Pulmonary closure ~150 Hz, 300ms after S1)
        bt_s2 = bt + 0.30
        s2_mask = (t >= bt_s2) & (t <= bt_s2 + 0.10)
        t_s2 = t[s2_mask] - bt_s2
        signal[s2_mask] += 0.6 * np.exp(-t_s2 * 45) * np.sin(2 * np.pi * 150 * t_s2)

    # Add subtle background physiological noise
    signal += np.random.normal(0, 0.015, len(t))
    return signal

def generate_murmur_heart_sound(sr=SAMPLE_RATE, duration=DURATION, bpm=75, murmur_type="systolic"):
    """Generates cardiac murmur acoustic waveforms (systolic or diastolic turbulence)."""
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)
    
    beat_interval = 60.0 / bpm
    beat_times = np.arange(0.2, duration - 0.4, beat_interval)
    
    for bt in beat_times:
        bt += np.random.uniform(-0.015, 0.015)
        
        # S1 Beat
        s1_mask = (t >= bt) & (t <= bt + 0.12)
        t_s1 = t[s1_mask] - bt
        signal[s1_mask] += 0.7 * np.exp(-t_s1 * 35) * np.sin(2 * np.pi * 115 * t_s1)
        
        bt_s2 = bt + 0.30
        
        if murmur_type == "systolic":
            # Systolic Murmur (Turbulence band between S1 and S2: 300-600 Hz)
            murmur_mask = (t > bt + 0.06) & (t < bt_s2 - 0.02)
            t_m = t[murmur_mask] - (bt + 0.06)
            # Diamond crescendo-decrescendo envelope
            env = np.sin(np.pi * t_m / (bt_s2 - 0.08 - bt)) ** 2
            noise_band = np.random.normal(0, 1, len(t_m)) * np.sin(2 * np.pi * 450 * t_m)
            signal[murmur_mask] += 0.45 * env * noise_band
            
        elif murmur_type == "diastolic":
            # Diastolic Murmur (Decrescendo rumble after S2)
            murmur_mask = (t > bt_s2 + 0.05) & (t < bt_s2 + 0.35)
            t_m = t[murmur_mask] - (bt_s2 + 0.05)
            env = np.exp(-t_m * 8)
            noise_band = np.random.normal(0, 1, len(t_m)) * np.sin(2 * np.pi * 220 * t_m)
            signal[murmur_mask] += 0.35 * env * noise_band
            
        # S2 Beat
        s2_mask = (t >= bt_s2) & (t <= bt_s2 + 0.10)
        t_s2 = t[s2_mask] - bt_s2
        signal[s2_mask] += 0.55 * np.exp(-t_s2 * 45) * np.sin(2 * np.pi * 145 * t_s2)

    signal += np.random.normal(0, 0.02, len(t))
    return signal

def generate_extrastole_heart_sound(sr=SAMPLE_RATE, duration=DURATION):
    """Generates premature contraction (extrasystole) with compensatory pause."""
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)
    
    # Normal beats at 0.3s, 1.1s, then premature beat at 1.5s, then long pause to 2.7s, 3.5s, 4.3s
    beat_times = [0.3, 1.1, 1.5, 2.7, 3.5, 4.3]
    
    for bt in beat_times:
        s1_mask = (t >= bt) & (t <= bt + 0.12)
        t_s1 = t[s1_mask] - bt
        signal[s1_mask] += 0.8 * np.exp(-t_s1 * 35) * np.sin(2 * np.pi * 120 * t_s1)
        
        bt_s2 = bt + 0.28
        s2_mask = (t >= bt_s2) & (t <= bt_s2 + 0.10)
        t_s2 = t[s2_mask] - bt_s2
        signal[s2_mask] += 0.6 * np.exp(-t_s2 * 45) * np.sin(2 * np.pi * 150 * t_s2)

    signal += np.random.normal(0, 0.02, len(t))
    return signal

def generate_artifact_sound(sr=SAMPLE_RATE, duration=DURATION):
    """Generates non-cardiac friction/noise/ambient artifact recordings."""
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.random.normal(0, 0.15, len(t))
    
    # Add friction spikes
    num_spikes = np.random.randint(4, 10)
    for _ in range(num_spikes):
        idx = np.random.randint(0, len(t) - 500)
        spike = np.random.uniform(-0.8, 0.8, 500) * np.hanning(500)
        signal[idx:idx+500] += spike
        
    # Add 50Hz hum
    signal += 0.1 * np.sin(2 * np.pi * 50 * t)
    return signal

def generate_and_save_expanded_dataset(num_samples_per_class=300):
    print("\n" + "="*60)
    print("     EXPANDING HEART SOUND DATASET SUITE (SYNTHESIS + AUGMENTATION)    ")
    print("="*60)
    
    os.makedirs(os.path.join(DATASET_DIR, "normal"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "murmur"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "extrastole"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "artifact"), exist_ok=True)

    print(f"[*] Generating {num_samples_per_class} high-fidelity samples per category...")
    
    # 1. Normal Samples
    for i in range(num_samples_per_class):
        bpm = np.random.randint(58, 95)
        sig = generate_normal_heart_sound(bpm=bpm)
        sig = sig / (np.max(np.abs(sig)) + 1e-6) # normalize
        filename = f"synth_normal_{i:04d}.wav"
        wavfile.write(os.path.join(DATASET_DIR, "normal", filename), SAMPLE_RATE, (sig * 32767).astype(np.int16))

    print(f"[+] Generated {num_samples_per_class} Normal Heart Sound recordings.")

    # 2. Murmur Samples (Systolic + Diastolic)
    for i in range(num_samples_per_class):
        m_type = "systolic" if i % 2 == 0 else "diastolic"
        bpm = np.random.randint(62, 100)
        sig = generate_murmur_heart_sound(bpm=bpm, murmur_type=m_type)
        sig = sig / (np.max(np.abs(sig)) + 1e-6)
        filename = f"synth_murmur_{i:04d}.wav"
        wavfile.write(os.path.join(DATASET_DIR, "murmur", filename), SAMPLE_RATE, (sig * 32767).astype(np.int16))

    print(f"[+] Generated {num_samples_per_class} Cardiac Murmur recordings.")

    # 3. Extrasystole Samples
    for i in range(num_samples_per_class // 2):
        sig = generate_extrastole_heart_sound()
        sig = sig / (np.max(np.abs(sig)) + 1e-6)
        filename = f"synth_extrastole_{i:04d}.wav"
        wavfile.write(os.path.join(DATASET_DIR, "extrastole", filename), SAMPLE_RATE, (sig * 32767).astype(np.int16))

    print(f"[+] Generated {num_samples_per_class // 2} Extrasystole Heart Sound recordings.")

    # 4. Artifact Samples
    for i in range(num_samples_per_class // 2):
        sig = generate_artifact_sound()
        sig = sig / (np.max(np.abs(sig)) + 1e-6)
        filename = f"synth_artifact_{i:04d}.wav"
        wavfile.write(os.path.join(DATASET_DIR, "artifact", filename), SAMPLE_RATE, (sig * 32767).astype(np.int16))

    print(f"[+] Generated {num_samples_per_class // 2} Audio Artifact recordings.")

    # Summary
    total_files = 0
    for root, _, files in os.walk(DATASET_DIR):
        wavs = [f for f in files if f.endswith('.wav')]
        if wavs:
            folder = os.path.basename(root)
            print(f"  - Category '{folder}': {len(wavs)} total WAV files")
            total_files += len(wavs)

    print(f"\n[+] Expanded Heart Sound Dataset size: {total_files} total WAV files!")

if __name__ == "__main__":
    generate_and_save_expanded_dataset(num_samples_per_class=350)
