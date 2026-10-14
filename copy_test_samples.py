import os
import shutil

TEST_DIR = "C:/Users/Student/Desktop/Test_Audio_Samples"
SOURCE_DIR = "C:/Users/Student/Desktop/THEJAS/heart_sounds"

def prepare_desktop_test_samples():
    os.makedirs(TEST_DIR, exist_ok=True)
    print(f"[*] Creating easy test sample folder at: {TEST_DIR}\n")

    samples_to_copy = [
        # (category, source_filename, new_friendly_name)
        ('normal', 'synth_normal_0001.wav', '1_Normal_Heartbeat_Clear_S1_S2.wav'),
        ('normal', 'synth_normal_0025.wav', '2_Normal_Heartbeat_72BPM.wav'),
        ('murmur', 'synth_murmur_0000.wav', '3_Abnormal_Systolic_Murmur.wav'),
        ('murmur', 'synth_murmur_0001.wav', '4_Abnormal_Diastolic_Murmur.wav'),
        ('artifact', 'synth_artifact_0000.wav', '5_Audio_Artifact_External_Noise.wav'),
        ('extrastole', 'synth_extrastole_0000.wav', '6_Extrasystole_Premature_Beat.wav'),
    ]

    copied_files = []
    for cat, src_name, new_name in samples_to_copy:
        src_path = os.path.join(SOURCE_DIR, cat, src_name)
        dst_path = os.path.join(TEST_DIR, new_name)
        
        # Fallback to any file in that category if specific name not found
        if not os.path.exists(src_path):
            cat_dir = os.path.join(SOURCE_DIR, cat)
            if os.path.exists(cat_dir):
                files = [f for f in os.listdir(cat_dir) if f.endswith('.wav')]
                if files:
                    src_path = os.path.join(cat_dir, files[0])

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_files.append((new_name, dst_path))
            print(f"[+] Copied test file: {new_name}")

    print(f"\n[+] Total {len(copied_files)} test WAV audio files ready in Desktop folder!")

if __name__ == "__main__":
    prepare_desktop_test_samples()
