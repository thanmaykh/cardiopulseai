import numpy as np
import scipy.signal as signal

def butter_bandpass_filter(data, lowcut=20.0, highcut=500.0, fs=16000, order=4):
    """
    Butterworth 4th-order bandpass filter.
    Keeps fundamental heart acoustic frequencies (20 Hz - 500 Hz),
    filtering out sub-audible breathing/rumble (<20 Hz) and high-frequency friction (>500 Hz).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

def notch_filter(data, freq=50.0, fs=16000, q=30.0):
    """
    IIR Notch Filter to remove 50 Hz / 60 Hz electrical mains hum.
    """
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = signal.iirnotch(w0, q)
    y = signal.filtfilt(b, a, data)
    return y

def spectral_subtraction_denoise(data, alpha=2.0, beta=0.02):
    """
    Spectral subtraction noise reduction algorithm.
    Estimates stationary background noise and subtracts it in frequency domain.
    """
    # Short-Time Fourier Transform
    f, t, Zxx = signal.stft(data, nperseg=512, noverlap=384)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # Estimate noise spectrum from first 10% frames
    noise_frames = max(1, int(Zxx.shape[1] * 0.10))
    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Subtract noise profile with over-subtraction factor alpha and spectral floor beta
    subtracted = magnitude - alpha * noise_profile
    cleaned_mag = np.where(subtracted > beta * noise_profile, subtracted, beta * noise_profile)
    
    # Reconstruct complex spectrum
    Zxx_cleaned = cleaned_mag * np.exp(1j * phase)
    
    # Inverse STFT
    _, cleaned_signal = signal.istft(Zxx_cleaned, nperseg=512, noverlap=384)
    
    # Fix length match
    if len(cleaned_signal) < len(data):
        cleaned_signal = np.pad(cleaned_signal, (0, len(data) - len(cleaned_signal)))
    else:
        cleaned_signal = cleaned_signal[:len(data)]
        
    return cleaned_signal

def clean_heart_sound_audio(audio_data, sr=16000, apply_denoise=True):
    """
    Complete Heart Sound Denoising & Enhancement Pipeline:
    1. 50Hz Notch Filter (Mains hum removal)
    2. 20Hz - 500Hz Bandpass Filter (Removes low rumble & high friction)
    3. Spectral Subtraction Denoising (Background noise reduction)
    4. Amplitude Normalization
    """
    cleaned = audio_data.copy()
    
    # 1. Remove 50 Hz & 60 Hz power line hum
    cleaned = notch_filter(cleaned, freq=50.0, fs=sr)
    cleaned = notch_filter(cleaned, freq=60.0, fs=sr)
    
    # 2. Apply Bandpass Filter (20Hz - 500Hz)
    cleaned = butter_bandpass_filter(cleaned, lowcut=20.0, highcut=500.0, fs=sr, order=4)
    
    # 3. Apply Spectral Subtraction if requested
    if apply_denoise:
        try:
            cleaned = spectral_subtraction_denoise(cleaned)
        except Exception as e:
            print(f"[!] Warning in spectral subtraction: {e}")

    # 4. Normalize Peak Amplitude
    max_val = np.max(np.abs(cleaned))
    if max_val > 0:
        cleaned = cleaned / max_val
        
    return cleaned
