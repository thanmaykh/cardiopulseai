import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import tensorflow as tf

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

from train_heart_sound_model import load_all_datasets, CLASSES, NUM_CLASSES, SEED

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def evaluate_and_generate_plots(model_path=None):
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "best_combined_heart_model.h5")
        
    print("\n" + "="*60)
    print("      HEART SOUND CLASSIFIER - MODEL EVALUATION SUITE      ")
    print("="*60)
    
    # 1. Load Dataset
    data_dirs = [
        os.path.join(BASE_DIR, "heart_sounds_merged"),
        os.path.join(BASE_DIR, "heart_sounds")
    ]
    print("[*] Loading and extracting test features...")
    X, y = load_all_datasets(data_dirs, apply_augmentation=False)
    
    if len(X) == 0:
        print("[!] Dataset missing, cannot evaluate.")
        return
        
    print(f"[+] Loaded {len(X)} evaluation samples with shape {X.shape}.")

    # 2. Train/Test Split (Same seed as training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 3. Load Trained Model
    if not os.path.exists(model_path):
        print(f"[!] Model file not found at: {model_path}")
        return
        
    print(f"[*] Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # 4. Predictions
    print("[*] Computing predictions on test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[+] Overall Test Accuracy: {acc * 100:.2f}%\n")
    
    print("Detailed Classification Report:")
    report_str = classification_report(y_test, y_pred, target_names=CLASSES)
    print(report_str)

    # Save report to text file
    report_path = os.path.join(BASE_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Overall Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
    print(f"[+] Saved evaluation report to '{report_path}'")

    # 5. Plot & Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Heart Sound Classification - Confusion Matrix', fontsize=14, pad=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix saved to '{cm_path}'")

    print("\n[+] Model Evaluation Completed Successfully!")

if __name__ == "__main__":
    evaluate_and_generate_plots()
