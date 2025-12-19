# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 🧠 Phase 4: Subvocalization Classification
# **Single-Channel sEMG → Silent Speech Recognition**
#
# Transfer Learning: Mouthing (L3) → Subvocal (L4)
#
# ---
# **Author:** Carl Kho | **Date:** December 2025 | **GPU:** A100 Recommended

# %% [markdown]
# ## 1️⃣ Setup & Data Upload

# %%
# Check GPU
!nvidia-smi

# %%
# Install dependencies
!pip install -q pandas numpy matplotlib seaborn scikit-learn tensorflow scipy

# %%
# Upload your data.zip (contains CSV files)
from google.colab import files
import zipfile
import os

print("📁 Upload your speech-capture.zip file:")
uploaded = files.upload()

# Extract
zip_name = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall('data')

print("\n✅ Extracted files:")
!ls -la data/

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(1738)
tf.random.set_seed(1738)

print(f"TensorFlow: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# %% [markdown]
# ## 2️⃣ Load Data

# %%
# Find the data directory (handles nested extraction)
import glob

csv_files = glob.glob('data/**/*.csv', recursive=True)
if not csv_files:
    csv_files = glob.glob('data/*.csv')

DATA_DIR = os.path.dirname(csv_files[0]) if csv_files else 'data'
print(f"Data directory: {DATA_DIR}")
print(f"Found CSVs: {csv_files}")

# %%
def load_spectrum_data(data_dir):
    """Load all motor intensity spectrum CSV files."""
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

# %%
# Quick look at the data
print("\n📊 Sample data (mouthing):")
data['mouthing'].head()

# %% [markdown]
# ## 📋 Due Diligence: Data Quality Assessment
#
# Before modeling, we perform comprehensive diagnostics to verify data quality,
# class balance, and signal characteristics.

# %%
# ==========================================
# 📋 DUE DILIGENCE: Comprehensive Diagnostics
# ==========================================
print("=" * 60)
print("📋 DUE DILIGENCE: Data Quality Assessment")
print("=" * 60)

# 1. Class Balance Check
print("\n1️⃣ CLASS BALANCE:")
for level_name, df in data.items():
    print(f"\n  {level_name.upper()}:")
    class_counts = df['Label'].value_counts()
    total = len(df)
    for label, count in class_counts.items():
        pct = count / total * 100
        print(f"    {label}: {count:,} samples ({pct:.1f}%)")

# %%
# 2. Signal Statistics Comparison
print("\n2️⃣ SIGNAL STATISTICS (Raw ADC values):")
stats_data = []
for level_name, df in data.items():
    stats = {
        'Level': level_name.upper(),
        'Mean': df['RawValue'].mean(),
        'Std': df['RawValue'].std(),
        'Min': df['RawValue'].min(),
        'Max': df['RawValue'].max(),
        'Range': df['RawValue'].max() - df['RawValue'].min()
    }
    stats_data.append(stats)

stats_df = pd.DataFrame(stats_data)
print(stats_df.to_string(index=False))

# %%
# 3. Amplitude Comparison Across Levels (Visual)
print("\n3️⃣ AMPLITUDE COMPARISON ACROSS MOTOR INTENSITY LEVELS:")
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
levels = ['overt', 'whisper', 'mouthing', 'subvocal', 'imagined']
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#95a5a6']

for ax, level, color in zip(axes, levels, colors):
    if level in data:
        # Take a 3-second sample
        sample = data[level]['RawValue'].values[:3000]
        ax.plot(sample, linewidth=0.5, color=color)
        ax.set_title(f'{level.upper()}\n(n={len(data[level]):,})', fontweight='bold')
        ax.set_xlabel('Samples')

axes[0].set_ylabel('Raw ADC Value')
plt.suptitle('Signal Amplitude Across Motor Intensity Levels', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('viz_amplitude_comparison.png', dpi=150)
plt.show()

# %%
# 4. Sample Duration Distribution
print("\n4️⃣ SAMPLE DURATION PER WORD (Block Lengths):")
for level_name in ['mouthing', 'subvocal']:
    if level_name in data:
        df = data[level_name].copy()
        df['label_change'] = df['Label'] != df['Label'].shift(1)
        df['block_id'] = df['label_change'].cumsum()
        block_lengths = df.groupby('block_id').size()

        print(f"\n  {level_name.upper()}:")
        print(f"    Mean block length: {block_lengths.mean():.0f} samples ({block_lengths.mean()/1000:.2f}s)")
        print(f"    Std: {block_lengths.std():.0f} samples")
        print(f"    Min: {block_lengths.min()} | Max: {block_lengths.max()}")
        print(f"    Total blocks: {len(block_lengths)}")

# %%
# 5. ADC Value Distribution
print("\n5️⃣ ADC VALUE DISTRIBUTION (Histogram):")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mouthing
axes[0].hist(data['mouthing']['RawValue'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_title('Mouthing (L3) ADC Distribution', fontweight='bold')
axes[0].set_xlabel('ADC Value')
axes[0].set_ylabel('Frequency')
axes[0].axvline(data['mouthing']['RawValue'].mean(), color='red', linestyle='--', label=f"Mean: {data['mouthing']['RawValue'].mean():.0f}")
axes[0].legend()

# Subvocal
axes[1].hist(data['subvocal']['RawValue'], bins=50, color='coral', alpha=0.7, edgecolor='black')
axes[1].set_title('Subvocal (L4) ADC Distribution', fontweight='bold')
axes[1].set_xlabel('ADC Value')
axes[1].set_ylabel('Frequency')
axes[1].axvline(data['subvocal']['RawValue'].mean(), color='red', linestyle='--', label=f"Mean: {data['subvocal']['RawValue'].mean():.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig('viz_adc_distribution.png', dpi=150)
plt.show()

# %%
# 6. Per-Class Signal Comparison (Mouthing)
print("\n6️⃣ PER-CLASS STATISTICS (Mouthing):")
for label in data['mouthing']['Label'].unique():
    subset = data['mouthing'][data['mouthing']['Label'] == label]['RawValue']
    print(f"  {label}: mean={subset.mean():.1f}, std={subset.std():.1f}, range=[{subset.min()}, {subset.max()}]")

print("\n" + "=" * 60)
print("✅ DUE DILIGENCE COMPLETE")
print("=" * 60)

# %% [markdown]
# ## 3️⃣ Preprocessing & Windowing

# %%
from scipy.signal import butter, filtfilt, iirnotch

def bandpass_filter(signal, fs=1000, lowcut=1.0, highcut=45.0, order=4):
    """Apply Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, fs=1000, freq=60.0, Q=30.0):
    """Remove 60Hz power line noise."""
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal)

def preprocess_signal(signal):
    """Full preprocessing pipeline."""
    signal = bandpass_filter(signal)
    signal = notch_filter(signal)
    # Z-score normalize
    return (signal - signal.mean()) / (signal.std() + 1e-8)

# %%
def create_windows(df, window_size=1000, center_offset=1000):
    """
    Create fixed-size windows grouped by label transitions.

    IMPORTANT: Words were vocalized at countdown "2" (middle of 3-sec window).
    So we extract samples [1000:2000] to capture the actual articulation.

    Args:
        window_size: Size of window to extract (1000 = 1 second @ 1000Hz)
        center_offset: Where the word starts in the raw block (1000 = at 1 second)
    """
    windows_X = []
    windows_y = []

    # Find label transitions
    df['label_change'] = df['Label'] != df['Label'].shift(1)
    df['block_id'] = df['label_change'].cumsum()

    for block_id, block in df.groupby('block_id'):
        label = block['Label'].iloc[0]
        signal = block['RawValue'].values

        # Extract MIDDLE portion where word was actually spoken
        # Words vocalized at countdown "2" = samples 1000-2000
        if len(signal) >= center_offset + window_size:
            signal = signal[center_offset:center_offset + window_size]
        elif len(signal) >= window_size:
            # Fallback: take last window_size samples
            signal = signal[-window_size:]
        else:
            # Pad if too short
            pad_size = window_size - len(signal)
            signal = np.pad(signal, (0, pad_size), mode='mean')

        # Preprocess
        try:
            signal = preprocess_signal(signal)
            windows_X.append(signal.reshape(-1, 1))
            windows_y.append(label)
        except:
            continue  # Skip problematic windows

    return np.array(windows_X), np.array(windows_y)

print("Creating windows (extracting MIDDLE 1-second where word was spoken)...")
X_mouthing, y_mouthing = create_windows(data['mouthing'])
X_subvocal, y_subvocal = create_windows(data['subvocal'])

print(f"\n📊 Mouthing (L3 - Training): {X_mouthing.shape}")
print(f"📊 Subvocal (L4 - Testing): {X_subvocal.shape}")

# %%
# 📊 Visualize random samples for each class
print("\n📊 Random samples per class (Mouthing - L3):")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for i, cls in enumerate(le.classes_):
    # Find indices for this class
    cls_indices = np.where(y_mouthing == cls)[0]
    if len(cls_indices) > 0:
        # Pick random sample
        rand_idx = np.random.choice(cls_indices)
        signal = X_mouthing[rand_idx].flatten()

        axes[i].plot(signal, linewidth=0.8, color='steelblue')
        axes[i].set_title(f'{cls} (sample #{rand_idx})', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Samples (1000 = 1 second)')
        axes[i].set_ylabel('Normalized Amplitude')
        axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('Random Signal Samples per Word Class (Mouthing - L3)', fontsize=14)
plt.tight_layout()
plt.savefig('viz_random_samples_mouthing.png', dpi=150)
plt.show()

# Same for subvocal
print("\n📊 Random samples per class (Subvocal - L4):")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for i, cls in enumerate(le.classes_):
    cls_indices = np.where(y_subvocal == cls)[0]
    if len(cls_indices) > 0:
        rand_idx = np.random.choice(cls_indices)
        signal = X_subvocal[rand_idx].flatten()

        axes[i].plot(signal, linewidth=0.8, color='coral')
        axes[i].set_title(f'{cls} (sample #{rand_idx})', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Samples (1000 = 1 second)')
        axes[i].set_ylabel('Normalized Amplitude')
        axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('Random Signal Samples per Word Class (Subvocal - L4)', fontsize=14)
plt.tight_layout()
plt.savefig('viz_random_samples_subvocal.png', dpi=150)
plt.show()

# %%
# Encode labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(np.concatenate([y_mouthing, y_subvocal]))

y_mouthing_enc = le.transform(y_mouthing)
y_subvocal_enc = le.transform(y_subvocal)

print(f"Classes: {le.classes_}")
print(f"Label mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

# %%
# Train/Val split on mouthing (source domain)
X_train, X_val, y_train, y_val = train_test_split(
    X_mouthing, y_mouthing_enc,
    test_size=0.15,
    random_state=42,
    stratify=y_mouthing_enc
)

# Test set is subvocal (target domain)
X_test, y_test = X_subvocal, y_subvocal_enc

print(f"\n📊 Train: {X_train.shape}")
print(f"📊 Val: {X_val.shape}")
print(f"📊 Test (L4): {X_test.shape}")

# %% [markdown]
# ## 4️⃣ Feature Extraction (EXTENDED for better accuracy)

# %%
# ACCURACY IMPROVEMENT #1: Extended Feature Set
def extract_features_extended(X):
    """
    Extract EXTENDED features for better discrimination.

    Features:
    - Time domain: MAV, ZCR, SD, MAX, RMS, Waveform Length
    - Temporal: Energy in 4 quarters
    - Frequency: Dominant frequency, spectral centroid

    Total: 14 features per window
    """
    from scipy.fft import fft

    features = []
    for window in X:
        signal = window.flatten()
        n = len(signal)

        # Time domain features
        mav = np.mean(np.abs(signal))
        zcr = np.sum(np.diff(np.sign(signal)) != 0)
        sd = np.std(signal)
        max_amp = np.max(np.abs(signal))
        rms = np.sqrt(np.mean(signal**2))
        waveform_length = np.sum(np.abs(np.diff(signal)))

        # Temporal features (quarters)
        e1 = np.mean(np.abs(signal[:n//4]))
        e2 = np.mean(np.abs(signal[n//4:n//2]))
        e3 = np.mean(np.abs(signal[n//2:3*n//4]))
        e4 = np.mean(np.abs(signal[3*n//4:]))

        # Frequency features (FFT)
        fft_vals = np.abs(fft(signal))[:n//2]  # Only positive frequencies
        freqs = np.linspace(0, 500, n//2)  # 0-500Hz for 1000Hz sampling

        # Dominant frequency
        dom_freq_idx = np.argmax(fft_vals[1:]) + 1  # Skip DC
        dom_freq = freqs[dom_freq_idx]

        # Spectral centroid
        spectral_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-8)

        # Spectral energy in speech band (1-45Hz)
        speech_band_mask = (freqs >= 1) & (freqs <= 45)
        speech_band_energy = np.sum(fft_vals[speech_band_mask])

        features.append([
            mav, zcr, sd, max_amp, rms, waveform_length,
            e1, e2, e3, e4,
            dom_freq, spectral_centroid, speech_band_energy,
            e2 - e1  # Onset indicator
        ])

    return np.array(features)

print("Extracting EXTENDED features (14 features)...")
X_train_feat = extract_features_extended(X_train)
X_val_feat = extract_features_extended(X_val)
X_test_feat = extract_features_extended(X_test)

print(f"Feature shapes: Train {X_train_feat.shape}, Val {X_val_feat.shape}, Test {X_test_feat.shape}")

# %%
# ACCURACY IMPROVEMENT #2: Data Augmentation
def augment_data(X, y, factor=3):
    """
    Augment training data with jitter, scaling, and time shift.
    """
    X_aug = [X]
    y_aug = [y]

    for _ in range(factor - 1):
        X_new = []
        for window in X:
            aug = window.copy()

            # Random jitter
            aug += np.random.normal(0, 0.05, aug.shape)

            # Random scaling
            aug *= np.random.uniform(0.9, 1.1)

            # Random time shift (circular)
            shift = np.random.randint(-50, 50)
            aug = np.roll(aug, shift, axis=0)

            X_new.append(aug)

        X_aug.append(np.array(X_new))
        y_aug.append(y)

    return np.vstack(X_aug), np.hstack(y_aug)

print("\nAugmenting training data (3x)...")
X_train_aug, y_train_aug = augment_data(X_train, y_train, factor=3)
X_train_feat_aug = extract_features_extended(X_train_aug)
print(f"Augmented: {X_train_feat.shape} → {X_train_feat_aug.shape}")

# %% [markdown]
# ## 5️⃣ Random Forest Baseline

# %%
print("🌲 Training Random Forest (with AUGMENTED data)...")

rf = RandomForestClassifier(
    n_estimators=200,  # More trees for better accuracy
    max_depth=20,      # Prevent overfitting
    min_samples_split=5,
    random_state=1738,
    n_jobs=-1
)

# Train on AUGMENTED data
rf.fit(X_train_feat_aug, y_train_aug)

# Evaluate on source domain (val)
y_val_pred = rf.predict(X_val_feat)
val_acc = accuracy_score(y_val, y_val_pred)

# Evaluate on target domain (test = L4)
y_test_pred = rf.predict(X_test_feat)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n✅ Val Accuracy (L3): {val_acc:.4f}")
print(f"✅ Test Accuracy (L4): {test_acc:.4f}")
print(f"📉 Transfer Gap: {val_acc - test_acc:.4f}")

# %%
# Compare: Would non-augmented do better?
print("\n📊 Ablation: Augmented vs Non-Augmented:")
rf_no_aug = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=1738, n_jobs=-1)
rf_no_aug.fit(X_train_feat, y_train)
no_aug_acc = accuracy_score(y_test, rf_no_aug.predict(X_test_feat))
print(f"  Without augmentation: {no_aug_acc:.4f}")
print(f"  With augmentation:    {test_acc:.4f}")
print(f"  Improvement: {(test_acc - no_aug_acc)*100:+.2f}%")

# %%
# Classification Report
print("\n📊 Classification Report (Test - L4):")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# %%
# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_test_pred, normalize='true')
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Random Forest Confusion Matrix (Test Acc: {test_acc:.2%})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', dpi=150)
plt.show()

# %% [markdown]
# ## 🔬 SANITY CHECK: Same-Domain Classification
#
# Before attempting cross-domain transfer (L3→L4), let's verify that we can
# classify words **within the same domain**. This tells us: "Is the signal
# even classifiable with our hardware?"

# %%
# ==========================================
# 🔬 SANITY CHECK: Same-Domain Classification
# ==========================================
print("=" * 60)
print("🔬 SANITY CHECK: Train & Test on MOUTHING ONLY (same domain)")
print("=" * 60)

# Split mouthing data properly (train/test from same source)
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_mouthing, y_mouthing_enc,
    test_size=0.20,
    random_state=42,
    stratify=y_mouthing_enc
)

# Extract features
X_m_train_feat = extract_features(X_m_train)
X_m_test_feat = extract_features(X_m_test)

# Train RF on mouthing only
rf_sanity = RandomForestClassifier(n_estimators=100, random_state=1738, n_jobs=-1)
rf_sanity.fit(X_m_train_feat, y_m_train)

# Evaluate same-domain
y_m_pred = rf_sanity.predict(X_m_test_feat)
sanity_acc = accuracy_score(y_m_test, y_m_pred)

print(f"\n✅ Same-Domain Accuracy (L3→L3): {sanity_acc:.4f}")
print(f"   (This is the 'ceiling' - best we can do)")
print(f"\n📊 Classification Report (L3 only):")
print(classification_report(y_m_test, y_m_pred, target_names=le.classes_))

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_m_test, y_m_pred, normalize='true')
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Purples',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'SANITY CHECK: Mouthing Only (Acc: {sanity_acc:.2%})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('sanity_check_mouthing.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("INTERPRETATION:")
print("  > 70%: Signal is good! Transfer L3→L4 is the hard part")
print("  < 50%: Hardware/signal issue - words may not be distinguishable")
print("=" * 60)

# %% [markdown]
# ## 6️⃣ MaxCRNN (Deep Learning)

# %%
def inception_block(x, filters):
    """1D Inception block with parallel convolutions."""
    conv1 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
    conv3 = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
    conv5 = layers.Conv1D(filters, 5, padding='same', activation='relu')(x)
    pool = layers.MaxPooling1D(3, strides=1, padding='same')(x)
    pool = layers.Conv1D(filters, 1, padding='same', activation='relu')(pool)
    return layers.Concatenate()([conv1, conv3, conv5, pool])

def build_maxcrnn(input_shape, n_classes):
    """Build MaxCRNN: Inception + Bi-LSTM + Attention"""
    inputs = layers.Input(shape=input_shape)

    # Inception blocks
    x = inception_block(inputs, filters=32)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling1D(2)(x)

    x = inception_block(x, filters=64)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling1D(2)(x)

    # Bi-LSTM
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    # Multi-Head Attention
    x = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)

    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='MaxCRNN')

# Build model
n_classes = len(le.classes_)
model = build_maxcrnn(input_shape=(X_train.shape[1], 1), n_classes=n_classes)
model.summary()

# %%
# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
]

print("🚀 Training MaxCRNN...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# %%
# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train')
axes[0].plot(history.history['val_loss'], label='Val')
axes[0].set_title('Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train')
axes[1].plot(history.history['val_accuracy'], label='Val')
axes[1].set_title('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('maxcrnn_training_curves.png', dpi=150)
plt.show()

# %%
# Evaluate MaxCRNN
print("\n📊 MaxCRNN Evaluation:")

# Val (L3)
val_loss, val_acc_nn = model.evaluate(X_val, y_val, verbose=0)
print(f"Val Accuracy (L3): {val_acc_nn:.4f}")

# Test (L4)
test_loss, test_acc_nn = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy (L4): {test_acc_nn:.4f}")
print(f"Transfer Gap: {val_acc_nn - test_acc_nn:.4f}")

# %%
# MaxCRNN Confusion Matrix
y_test_pred_nn = np.argmax(model.predict(X_test), axis=1)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_test_pred_nn, normalize='true')
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'MaxCRNN Confusion Matrix (Test Acc: {test_acc_nn:.2%})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('maxcrnn_confusion_matrix.png', dpi=150)
plt.show()

print("\n📊 Classification Report (MaxCRNN - L4):")
print(classification_report(y_test, y_test_pred_nn, target_names=le.classes_))

# %% [markdown]
# ## 7️⃣ Model Comparison

# %%
# Compare models
rf_test_acc = accuracy_score(y_test, y_test_pred)
nn_test_acc = accuracy_score(y_test, y_test_pred_nn)

results = pd.DataFrame({
    'Model': ['Random Forest', 'MaxCRNN'],
    'Test Accuracy (L4)': [rf_test_acc, nn_test_acc],
    'Val Accuracy (L3)': [accuracy_score(y_val, rf.predict(X_val_feat)), val_acc_nn]
})

print("\n🏆 Model Comparison:")
print(results)

# Bar chart
plt.figure(figsize=(8, 5))
x = np.arange(2)
width = 0.35
plt.bar(x - width/2, results['Val Accuracy (L3)'], width, label='Val (L3)', color='steelblue')
plt.bar(x + width/2, results['Test Accuracy (L4)'], width, label='Test (L4)', color='coral')
plt.xticks(x, results['Model'])
plt.ylabel('Accuracy')
plt.title('Transfer Learning: L3 → L4')
plt.legend()
plt.ylim(0, 1)
for i, v in enumerate(results['Test Accuracy (L4)']):
    plt.text(i + width/2, v + 0.02, f'{v:.2%}', ha='center')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# %% [markdown]
# ## 8️⃣ Save Models & Download

# %%
# Save Random Forest
import pickle
with open('random_forest_phase4.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("✅ Saved: random_forest_phase4.pkl")

# Save MaxCRNN
model.save('maxcrnn_phase4.keras')
print("✅ Saved: maxcrnn_phase4.keras")

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✅ Saved: label_encoder.pkl")

# %%
# Download results
from google.colab import files

# Zip everything
!zip -r phase4_results.zip *.pkl *.keras *.png

files.download('phase4_results.zip')

# %% [markdown]
# ## 📊 Summary
#
# | Model | Val Acc (L3) | Test Acc (L4) | Transfer Gap |
# |-------|--------------|---------------|--------------|
# | Random Forest | TBD | TBD | TBD |
# | MaxCRNN | TBD | TBD | TBD |
#
# ---
# **Next Steps:**
# 1. Fill TBD values after running
# 2. Export confusion matrices for assignment
# 3. Analyze per-class performance

# %% [markdown]
# ---
# # 🚀 ADVANCED STRATEGIES FOR ACCURACY IMPROVEMENT
# ---

# %% [markdown]
# ## 9️⃣ Binary Classification: WORD vs REST
#
# Before attempting 4-class classification, let's see if we can at least
# distinguish "any word" from "rest". This is often easier and can give us
# confidence in the signal quality.

# %%
print("=" * 60)
print("🎯 STRATEGY 1: Binary Classification (WORD vs REST)")
print("=" * 60)

# Convert to binary labels
def to_binary(y, le):
    """Convert multi-class labels to binary (WORD vs REST)."""
    class_names = le.classes_
    rest_idx = np.where(class_names == 'REST')[0][0] if 'REST' in class_names else -1
    return (y != rest_idx).astype(int)  # 0=REST, 1=WORD

y_train_binary = to_binary(y_train, le)
y_val_binary = to_binary(y_val, le)
y_test_binary = to_binary(y_test, le)

print(f"Binary distribution (train): REST={np.sum(y_train_binary==0)}, WORD={np.sum(y_train_binary==1)}")

# Train binary RF
rf_binary = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=1738, n_jobs=-1)
rf_binary.fit(X_train_feat_aug, np.tile(y_train_binary, 3))  # Match augmented size

# Evaluate
y_test_pred_binary = rf_binary.predict(X_test_feat)
binary_acc = accuracy_score(y_test_binary, y_test_pred_binary)

print(f"\n✅ Binary Accuracy (WORD vs REST): {binary_acc:.4f}")
print(f"   (If this is high, signal is good but words are hard to distinguish)")

# Confusion matrix
plt.figure(figsize=(6, 5))
cm_binary = confusion_matrix(y_test_binary, y_test_pred_binary, normalize='true')
sns.heatmap(cm_binary, annot=True, fmt='.2f', cmap='Oranges',
            xticklabels=['REST', 'WORD'], yticklabels=['REST', 'WORD'])
plt.title(f'Binary Classification (Acc: {binary_acc:.2%})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('binary_confusion_matrix.png', dpi=150)
plt.show()

# %% [markdown]
# ## 🔟 Spectrogram + 2D CNN (ImageNet Transfer Learning)
#
# Convert 1D signals to mel-spectrograms and use MobileNetV2 pretrained on ImageNet.
# This leverages millions of image-trained weights for pattern recognition.

# %%
print("=" * 60)
print("🎯 STRATEGY 2: Spectrogram + MobileNetV2 Transfer Learning")
print("=" * 60)

import librosa
import librosa.display

def signal_to_spectrogram(signal, sr=1000, n_mels=64, target_size=(96, 96)):
    """
    Convert 1D EMG signal to mel-spectrogram image.
    """
    # Compute mel-spectrogram
    S = librosa.feature.melspectrogram(
        y=signal.astype(float),
        sr=sr,
        n_mels=n_mels,
        fmax=sr/2
    )
    # Convert to dB
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Normalize to 0-255
    S_norm = ((S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8) * 255).astype(np.uint8)

    # Resize to target size
    from scipy.ndimage import zoom
    zoom_factors = (target_size[0] / S_norm.shape[0], target_size[1] / S_norm.shape[1])
    S_resized = zoom(S_norm, zoom_factors, order=1)

    # Convert to 3-channel (RGB) for ImageNet models
    S_rgb = np.stack([S_resized, S_resized, S_resized], axis=-1)

    return S_rgb

# Create spectrogram dataset
print("\nConverting signals to spectrograms...")
X_train_spec = np.array([signal_to_spectrogram(x.flatten()) for x in X_train])
X_val_spec = np.array([signal_to_spectrogram(x.flatten()) for x in X_val])
X_test_spec = np.array([signal_to_spectrogram(x.flatten()) for x in X_test])

print(f"Spectrogram shapes: Train {X_train_spec.shape}, Val {X_val_spec.shape}, Test {X_test_spec.shape}")

# %%
# Visualize sample spectrograms
print("\n📊 Sample spectrograms per class:")
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for i, cls in enumerate(le.classes_):
    cls_idx = np.where(le.classes_ == cls)[0][0]
    sample_indices = np.where(y_train == cls_idx)[0]
    if len(sample_indices) > 0:
        idx = sample_indices[0]
        axes[i].imshow(X_train_spec[idx], aspect='auto', origin='lower')
        axes[i].set_title(f'{cls}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Mel Bins')

plt.suptitle('Mel-Spectrograms per Word Class', fontsize=14)
plt.tight_layout()
plt.savefig('viz_spectrograms.png', dpi=150)
plt.show()

# %%
# Build MobileNetV2 transfer learning model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

def build_spectrogram_cnn(input_shape=(96, 96, 3), n_classes=4):
    """
    MobileNetV2 with ImageNet weights for spectrogram classification.
    """
    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(base_model.input, outputs, name='SpecCNN_MobileNetV2')
    return model

print("\nBuilding MobileNetV2 spectrogram model...")
spec_model = build_spectrogram_cnn(n_classes=len(le.classes_))
spec_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(f"Total params: {spec_model.count_params():,}")

# %%
# Train spectrogram model
print("\n🚀 Training Spectrogram CNN...")
spec_callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

# Normalize spectrograms for ImageNet
X_train_spec_norm = X_train_spec / 255.0
X_val_spec_norm = X_val_spec / 255.0
X_test_spec_norm = X_test_spec / 255.0

spec_history = spec_model.fit(
    X_train_spec_norm, y_train,
    validation_data=(X_val_spec_norm, y_val),
    epochs=50,
    batch_size=16,
    callbacks=spec_callbacks,
    verbose=1
)

# %%
# Evaluate spectrogram model
print("\n📊 Spectrogram CNN Evaluation:")
val_loss, val_acc_spec = spec_model.evaluate(X_val_spec_norm, y_val, verbose=0)
test_loss, test_acc_spec = spec_model.evaluate(X_test_spec_norm, y_test, verbose=0)

print(f"Val Accuracy (L3): {val_acc_spec:.4f}")
print(f"Test Accuracy (L4): {test_acc_spec:.4f}")
print(f"Transfer Gap: {val_acc_spec - test_acc_spec:.4f}")

# Confusion matrix
y_test_pred_spec = np.argmax(spec_model.predict(X_test_spec_norm), axis=1)
plt.figure(figsize=(8, 6))
cm_spec = confusion_matrix(y_test, y_test_pred_spec, normalize='true')
sns.heatmap(cm_spec, annot=True, fmt='.2f', cmap='Reds',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Spectrogram CNN Confusion Matrix (Acc: {test_acc_spec:.2%})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('spectrogram_cnn_confusion.png', dpi=150)
plt.show()

# %% [markdown]
# ## 1️⃣1️⃣ Window Overlap Strategy
#
# Instead of non-overlapping windows, use 50% overlap to create more training samples.

# %%
print("=" * 60)
print("🎯 STRATEGY 3: Window Overlap (50%)")
print("=" * 60)

def create_windows_overlap(df, window_size=1000, overlap=0.5, center_offset=500):
    """
    Create overlapping windows for more training data.

    Args:
        window_size: Window size in samples
        overlap: Overlap fraction (0.5 = 50%)
        center_offset: Where word starts in block
    """
    step = int(window_size * (1 - overlap))
    windows_X = []
    windows_y = []

    df['label_change'] = df['Label'] != df['Label'].shift(1)
    df['block_id'] = df['label_change'].cumsum()

    for block_id, block in df.groupby('block_id'):
        label = block['Label'].iloc[0]
        signal = block['RawValue'].values

        # Slide window with overlap
        for start in range(0, max(1, len(signal) - window_size), step):
            window = signal[start:start + window_size]
            if len(window) == window_size:
                try:
                    window = preprocess_signal(window)
                    windows_X.append(window.reshape(-1, 1))
                    windows_y.append(label)
                except:
                    continue

    return np.array(windows_X), np.array(windows_y)

# Create overlapped windows
X_mouth_overlap, y_mouth_overlap = create_windows_overlap(data['mouthing'].copy())
y_mouth_overlap_enc = le.transform(y_mouth_overlap)

print(f"Original windows: {X_mouthing.shape[0]}")
print(f"Overlapped windows: {X_mouth_overlap.shape[0]} ({X_mouth_overlap.shape[0]/X_mouthing.shape[0]:.1f}x more)")

# Train/test split on overlapped data
X_train_ov, X_val_ov, y_train_ov, y_val_ov = train_test_split(
    X_mouth_overlap, y_mouth_overlap_enc, test_size=0.15, random_state=42, stratify=y_mouth_overlap_enc
)

# Extract features
X_train_ov_feat = extract_features_extended(X_train_ov)
X_val_ov_feat = extract_features_extended(X_val_ov)

# Train RF on overlapped data
rf_overlap = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=1738, n_jobs=-1)
rf_overlap.fit(X_train_ov_feat, y_train_ov)

# Evaluate on original test set
y_test_pred_ov = rf_overlap.predict(X_test_feat)
overlap_acc = accuracy_score(y_test, y_test_pred_ov)

print(f"\n✅ Overlap RF Accuracy (L4 test): {overlap_acc:.4f}")
print(f"   vs Original RF: {test_acc:.4f}")
print(f"   Improvement: {(overlap_acc - test_acc)*100:+.2f}%")

# %% [markdown]
# ## 📊 FINAL COMPARISON: All Strategies

# %%
print("\n" + "=" * 60)
print("📊 FINAL COMPARISON: All Strategies")
print("=" * 60)

final_results = pd.DataFrame({
    'Strategy': [
        'Random Forest (baseline)',
        'Random Forest (augmented + extended features)',
        'Binary (WORD vs REST)',
        'Spectrogram CNN (MobileNetV2)',
        'RF with 50% Window Overlap',
        'MaxCRNN'
    ],
    'Test Accuracy (L4)': [
        no_aug_acc,
        test_acc,
        binary_acc,
        test_acc_spec,
        overlap_acc,
        test_acc_nn
    ]
})

final_results = final_results.sort_values('Test Accuracy (L4)', ascending=False)
print(final_results.to_string(index=False))

# Bar chart
plt.figure(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(final_results)))
bars = plt.barh(final_results['Strategy'], final_results['Test Accuracy (L4)'], color=colors)
plt.xlabel('Test Accuracy (L4)')
plt.title('Accuracy Comparison: All Strategies')
plt.xlim(0, 1)
for bar, acc in zip(bars, final_results['Test Accuracy (L4)']):
    plt.text(acc + 0.01, bar.get_y() + bar.get_height()/2, f'{acc:.2%}', va='center')
plt.tight_layout()
plt.savefig('final_comparison.png', dpi=150)
plt.show()

# %% [markdown]
# ## 💾 Save All Models

# %%
# Save all models
print("\n💾 Saving all models...")

# RF models
with open('rf_augmented.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('rf_binary.pkl', 'wb') as f:
    pickle.dump(rf_binary, f)
with open('rf_overlap.pkl', 'wb') as f:
    pickle.dump(rf_overlap, f)

# Neural network models
model.save('maxcrnn_phase4.keras')
spec_model.save('spectrogram_cnn.keras')

# Label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("✅ All models saved!")

# %%
# Download all results
!pip install -q librosa  # Make sure librosa is in requirements
!zip -r phase4_all_results.zip *.pkl *.keras *.png

files.download('phase4_all_results.zip')

