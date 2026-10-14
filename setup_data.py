import os
import zipfile
import shutil

TARGET_DIR = "C:/Users/Student/Desktop/THEJAS/heart_sounds"
ZIP_PATH = "C:/Users/Student/Downloads/extracted_files/heart_sounds_dataset.zip"

def prepare_dataset():
    print(f"[*] Preparing unified heart sounds dataset in: {TARGET_DIR}")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    if os.path.exists(ZIP_PATH):
        print(f"[*] Extracting dataset from {ZIP_PATH}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall("C:/Users/Student/Desktop/THEJAS/temp_extract")
        
        source_inner = "C:/Users/Student/Desktop/THEJAS/temp_extract/Heartbeat-Sounds-Dataset/Heartbeat_Sound"
        if os.path.exists(source_inner):
            for folder in os.listdir(source_inner):
                src_folder = os.path.join(source_inner, folder)
                dst_folder = os.path.join(TARGET_DIR, folder)
                if os.path.isdir(src_folder):
                    os.makedirs(dst_folder, exist_ok=True)
                    for f in os.listdir(src_folder):
                        if f.endswith('.wav'):
                            shutil.copy2(os.path.join(src_folder, f), os.path.join(dst_folder, f))
            print(f"[+] Dataset successfully organized into {TARGET_DIR}!")
            shutil.rmtree("C:/Users/Student/Desktop/THEJAS/temp_extract", ignore_errors=True)
    
    # Audit total files in target dir
    total_count = 0
    for root, dirs, files in os.walk(TARGET_DIR):
        wav_files = [f for f in files if f.endswith('.wav')]
        if wav_files:
            folder_name = os.path.basename(root)
            print(f"  - Folder '{folder_name}': {len(wav_files)} WAV files")
            total_count += len(wav_files)
            
    print(f"[+] Total audio files in dataset: {total_count}")

if __name__ == "__main__":
    prepare_dataset()
