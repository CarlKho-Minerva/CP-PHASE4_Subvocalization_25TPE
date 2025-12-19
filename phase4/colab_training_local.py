"""
Phase 4: Subvocalization Classification - LOCAL VERSION
Run this locally with: python colab_training_local.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.signal import butter, filtfilt, iirnotch
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(1738)

print("=" * 60)
print("🧠 Phase 4: Subvocalization Classification (LOCAL)")
print("=" * 60)

# ===== CONFIGURATION =====
DATA_DIR = "phase4/speech-capture"  # Adjust path as needed

# ===== LOAD DATA =====
def load_spectrum_data(data_dir):
    files = {
        'overt': 'overt_data.csv',
        'whisper': 'whisper_data.csv',
        'mouthing': 'mouthing_data.csv',
        'subvocal': 'subvocal_data.csv',
        'imagined': 'imagined_data.csv'
    }
    data = {}
    for level, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            data[level] = df
            print(f"✅ Loaded {level}: {len(df):,} samples")
        else:
            print(f"⚠️  Missing: {filename}")
    return data

data = load_spectrum_data(DATA_DIR)

# ===== PREPROCESSING =====
def bandpass_filter(signal, fs=1000, lowcut=1.0, highcut=45.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, fs=1000, freq=60.0, Q=30.0):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal)

def preprocess_signal(signal):
    signal = bandpass_filter(signal)
    signal = notch_filter(signal)
    return (signal - signal.mean()) / (signal.std() + 1e-8)

# ===== WINDOWING =====
def create_windows(df, window_size=1000, center_offset=1000):
    """Extract MIDDLE 1-second where word was spoken at countdown '2'."""
    windows_X = []
    windows_y = []

    df = df.copy()
    df['label_change'] = df['Label'] != df['Label'].shift(1)
    df['block_id'] = df['label_change'].cumsum()

    for block_id, block in df.groupby('block_id'):
        label = block['Label'].iloc[0]
        signal = block['RawValue'].values

        if len(signal) >= center_offset + window_size:
            signal = signal[center_offset:center_offset + window_size]
        elif len(signal) >= window_size:
            signal = signal[-window_size:]
        else:
            pad_size = window_size - len(signal)
            signal = np.pad(signal, (0, pad_size), mode='mean')

        try:
            signal = preprocess_signal(signal)
            windows_X.append(signal.reshape(-1, 1))
            windows_y.append(label)
        except:
            continue

    return np.array(windows_X), np.array(windows_y)

print("\nCreating windows...")
X_mouthing, y_mouthing = create_windows(data['mouthing'])
X_subvocal, y_subvocal = create_windows(data['subvocal'])
print(f"📊 Mouthing: {X_mouthing.shape}, Subvocal: {X_subvocal.shape}")

# ===== ENCODE LABELS =====
le = LabelEncoder()
le.fit(np.concatenate([y_mouthing, y_subvocal]))
y_mouthing_enc = le.transform(y_mouthing)
y_subvocal_enc = le.transform(y_subvocal)
print(f"Classes: {le.classes_}")

# ===== TRAIN/TEST SPLIT =====
X_train, X_val, y_train, y_val = train_test_split(
    X_mouthing, y_mouthing_enc, test_size=0.15, random_state=42, stratify=y_mouthing_enc
)
X_test, y_test = X_subvocal, y_subvocal_enc

# ===== FEATURE EXTRACTION =====
def extract_features(X):
    features = []
    for window in X:
        signal = window.flatten()
        n = len(signal)

        mav = np.mean(np.abs(signal))
        zcr = np.sum(np.diff(np.sign(signal)) != 0)
        sd = np.std(signal)
        max_amp = np.max(np.abs(signal))
        rms = np.sqrt(np.mean(signal**2))
        wl = np.sum(np.abs(np.diff(signal)))

        e1 = np.mean(np.abs(signal[:n//4]))
        e2 = np.mean(np.abs(signal[n//4:n//2]))
        e3 = np.mean(np.abs(signal[n//2:3*n//4]))
        e4 = np.mean(np.abs(signal[3*n//4:]))

        fft_vals = np.abs(fft(signal))[:n//2]
        freqs = np.linspace(0, 500, n//2)
        dom_freq = freqs[np.argmax(fft_vals[1:]) + 1]
        spec_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-8)

        features.append([mav, zcr, sd, max_amp, rms, wl, e1, e2, e3, e4, dom_freq, spec_centroid, e2-e1])

    return np.array(features)

print("\nExtracting features...")
X_train_feat = extract_features(X_train)
X_val_feat = extract_features(X_val)
X_test_feat = extract_features(X_test)

# ===== RANDOM FOREST =====
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=1738, n_jobs=-1)
rf.fit(X_train_feat, y_train)

y_val_pred = rf.predict(X_val_feat)
y_test_pred = rf.predict(X_test_feat)

val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n✅ Val Accuracy (L3): {val_acc:.4f}")
print(f"✅ Test Accuracy (L4): {test_acc:.4f}")
print(f"📉 Transfer Gap: {val_acc - test_acc:.4f}")

# ===== SANITY CHECK: Binary =====
print("\n" + "=" * 60)
print("🎯 Binary Classification (WORD vs REST)")

rest_idx = np.where(le.classes_ == 'REST')[0][0] if 'REST' in le.classes_ else -1
y_test_binary = (y_test != rest_idx).astype(int)
y_pred_binary = (y_test_pred != rest_idx).astype(int)
binary_acc = accuracy_score(y_test_binary, y_pred_binary)
print(f"Binary Accuracy: {binary_acc:.4f}")

# ===== CONFUSION MATRIX =====
print("\n📊 Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_test_pred, normalize='true')
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix (Test Acc: {test_acc:.2%})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix_local.png', dpi=150)
plt.show()

print("\n✅ Done! Saved: confusion_matrix_local.png")
