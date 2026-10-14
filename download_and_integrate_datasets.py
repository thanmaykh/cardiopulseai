import os
import sys
import glob
import urllib.request
import zipfile
import shutil
import pandas as pd
import numpy as np

BASE_DIR = "C:/Users/Student/Desktop/THEJAS"
PHYSIONET_DIR = os.path.join(BASE_DIR, "physionet_datasets")
TARGET_DATASET_DIR = os.path.join(BASE_DIR, "heart_sounds_merged")

CLASSES = ['artifact', 'murmur_abnormal', 'normal']

# PhysioNet 2016 Challenge direct zip download URLs
PHYSIONET_2016_GET_ZIP_URL = "https://physionet.org/content/challenge-2016/get-zip/1.0.0/"

def setup_directories():
    os.makedirs(PHYSIONET_DIR, exist_ok=True)
    os.makedirs(TARGET_DATASET_DIR, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(TARGET_DATASET_DIR, c), exist_ok=True)
    print(f"[+] Output dataset folder ready: {TARGET_DATASET_DIR}")

def download_file(url, target_path):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
        print(f"  [+] Already downloaded: {os.path.basename(target_path)} ({os.path.getsize(target_path) / (1024*1024):.1f} MB)")
        return True
    print(f"  [*] Downloading dataset from {url} ...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=300) as resp, open(target_path, 'wb') as out_file:
            # Chunked download with progress reporting
            chunk_size = 1024 * 1024 # 1MB
            total_read = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                total_read += len(chunk)
                if total_read % (50 * 1024 * 1024) == 0:
                    print(f"    [->] Downloaded {total_read / (1024*1024):.1f} MB...", flush=True)
        print(f"  [+] Download complete! Saved to {os.path.basename(target_path)}")
        return True
    except Exception as e:
        print(f"  [!] Failed download: {e}")
        return False

def extract_zip(zip_path, extract_to):
    print(f"  [*] Extracting {os.path.basename(zip_path)}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"  [+] Extracted successfully!")
        return True
    except Exception as e:
        print(f"  [!] Failed to extract {zip_path}: {e}")
        return False

def process_physionet_2016():
    print("\n" + "="*60)
    print("      PROCESSING PHYSIONET/CinC CHALLENGE 2016 DATASET     ")
    print("="*60)
    
    zip_path = os.path.join(PHYSIONET_DIR, "challenge_2016_full.zip")
    extract_folder = os.path.join(PHYSIONET_DIR, "extracted")
    
    if not os.path.exists(extract_folder):
        if download_file(PHYSIONET_2016_GET_ZIP_URL, zip_path):
            extract_zip(zip_path, extract_folder)
            
    total_physionet_copied = 0
    ref_files = glob.glob(os.path.join(extract_folder, "**", "REFERENCE*.csv"), recursive=True)
    print(f"[*] Found {len(ref_files)} REFERENCE annotation CSV files.")
    
    for ref_csv in ref_files:
        try:
            ref_dir = os.path.dirname(ref_csv)
            folder_tag = os.path.basename(ref_dir)
            df = pd.read_csv(ref_csv, header=None)
            for _, row in df.iterrows():
                fname = str(row[0]).strip()
                if not fname.endswith('.wav'):
                    fname_wav = fname + '.wav'
                else:
                    fname_wav = fname
                
                src_wav = os.path.join(ref_dir, fname_wav)
                if not os.path.exists(src_wav):
                    continue
                    
                label_val = int(row[1])
                # -1 -> normal, 1 -> abnormal (murmur_abnormal)
                target_sub = 'normal' if label_val == -1 else 'murmur_abnormal'
                dst_wav = os.path.join(TARGET_DATASET_DIR, target_sub, f"pn2016_{folder_tag}_{fname_wav}")
                
                shutil.copy2(src_wav, dst_wav)
                total_physionet_copied += 1
        except Exception as e:
            print(f"  [!] Error processing {ref_csv}: {e}")

    print(f"[+] Total PhysioNet 2016 recordings processed & merged: {total_physionet_copied}")

def process_existing_kaggle_and_synth():
    print("\n" + "="*60)
    print("      MERGING KAGGLE + PASCAL + SYNTHETIC HEART SOUNDS     ")
    print("="*60)
    
    local_src_dir = os.path.join(BASE_DIR, "heart_sounds")
    if not os.path.exists(local_src_dir):
        print(f"[!] {local_src_dir} not found.")
        return
        
    copied = 0
    label_map = {
        'artifact': 'artifact',
        'extrahls': 'murmur_abnormal',
        'extrastole': 'murmur_abnormal',
        'murmur': 'murmur_abnormal',
        'normal': 'normal'
    }
    
    for root, _, files in os.walk(local_src_dir):
        folder_name = os.path.basename(root).lower()
        if folder_name not in label_map:
            continue
            
        target_sub = label_map[folder_name]
        for f in files:
            if f.endswith('.wav'):
                src_wav = os.path.join(root, f)
                dst_wav = os.path.join(TARGET_DATASET_DIR, target_sub, f"kaggle_{folder_name}_{f}")
                shutil.copy2(src_wav, dst_wav)
                copied += 1

    print(f"[+] Total local Kaggle/PASCAL/Synth recordings merged: {copied}")

def print_summary():
    print("\n" + "="*60)
    print("               FINAL MERGED DATASET SUMMARY               ")
    print("="*60)
    total = 0
    for c in CLASSES:
        c_dir = os.path.join(TARGET_DATASET_DIR, c)
        count = len(glob.glob(os.path.join(c_dir, "*.wav")))
        total += count
        print(f"  - Category '{c}': {count} WAV files")
    print(f"\n[+] Total Unified Dataset Size: {total} audio recordings!")

if __name__ == "__main__":
    setup_directories()
    process_physionet_2016()
    process_existing_kaggle_and_synth()
    print_summary()
