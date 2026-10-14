import os
import urllib.request
import zipfile
import pandas as pd

BASE_DIR = "C:/Users/Student/Desktop/THEJAS/physionet_2016"

DATASET_URLS = {
    "training-a": "https://physionet.org/content/challenge-2016/1.0.0/training-a.zip",
    "training-b": "https://physionet.org/content/challenge-2016/1.0.0/training-b.zip",
    "training-c": "https://physionet.org/content/challenge-2016/1.0.0/training-c.zip",
    "training-d": "https://physionet.org/content/challenge-2016/1.0.0/training-d.zip",
    "training-e": "https://physionet.org/content/challenge-2016/1.0.0/training-e.zip",
    "training-f": "https://physionet.org/content/challenge-2016/1.0.0/training-f.zip",
}

def download_and_extract():
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"[*] Target Directory: {BASE_DIR}")
    
    # Download training-a as primary extension dataset
    key = "training-a"
    url = DATASET_URLS[key]
    zip_path = os.path.join(BASE_DIR, f"{key}.zip")
    extract_path = os.path.join(BASE_DIR, key)
    
    if not os.path.exists(extract_path):
        print(f"[*] Downloading {key} from PhysioNet...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"[+] Downloaded {key}. Unzipping...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(BASE_DIR)
            print(f"[+] Extracted {key} successfully!")
        except Exception as e:
            print(f"[!] Error downloading {key}: {e}")
    else:
        print(f"[+] {key} directory already exists!")

if __name__ == "__main__":
    download_and_extract()
