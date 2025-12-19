# Section 3: Preprocessing, Cleaning & EDA

## Overview

Raw sEMG signals require preprocessing to remove noise and extract meaningful features. This section covers the signal processing pipeline and exploratory data analysis for single-channel subvocalization data.

## Signal Processing Pipeline

```
Raw ADC → Bandpass 1-45Hz → Notch 60Hz → Normalization → Epoch → Features
```

> **[INSERT IMAGE]** `images/viz_signal_pipeline.png`
> *Caption: Signal processing stages showing raw ADC, filtered, and normalized waveforms.*

> **Note:** The AD8232's hardware bandpass filter (0.5-40Hz) already provides substantial filtering aligned with AlterEgo's target range (1.3-50Hz). Software filters are applied for consistency and to remove residual power line interference.

## Preprocessing Steps

### 1. Bandpass Filtering

```python
from scipy.signal import butter, filtfilt, iirnotch
import numpy as np

def bandpass_filter(signal: np.ndarray,
                    fs: int = 1000,
                    lowcut: float = 1.0,
                    highcut: float = 45.0,
                    order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.

    Args:
        signal: Raw EMG signal (1D array)
        fs: Sampling frequency (1000Hz)
        lowcut: Lower cutoff frequency (1Hz - remove DC drift)
        highcut: Upper cutoff frequency (45Hz - below Nyquist)
        order: Filter order
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def notch_filter(signal: np.ndarray,
                 fs: int = 1000,
                 freq: float = 60.0,
                 Q: float = 30.0) -> np.ndarray:
    """
    Apply notch filter to remove power line interference.

    Args:
        signal: Bandpassed signal
        fs: Sampling frequency
        freq: Notch frequency (60Hz for Taiwan/US, 50Hz for EU)
        Q: Quality factor (higher = narrower notch)
    """
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal)
```

### 2. Normalization

```python
def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Min-Max normalization to [0, 1] range.
    """
    return (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)


def z_score_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Z-score normalization (zero mean, unit variance).
    Preferred for neural networks.
    """
    return (signal - signal.mean()) / (signal.std() + 1e-8)
```

### 3. Full Preprocessing Pipeline

```python
def preprocess_window(window: np.ndarray, fs: int = 1000) -> np.ndarray:
    """
    Apply full preprocessing pipeline to a single-channel window.

    Args:
        window: Raw ADC values, shape (window_size, 1) or (window_size,)
        fs: Sampling frequency

    Returns:
        Preprocessed signal, same shape as input
    """
    signal = window.flatten()

    # 1. Bandpass filter (1-45Hz)
    signal = bandpass_filter(signal, fs=fs, lowcut=1.0, highcut=45.0)

    # 2. Notch filter (60Hz - Taiwan power line)
    signal = notch_filter(signal, fs=fs, freq=60.0)

    # 3. Normalize
    signal = z_score_normalize(signal)

    return signal.reshape(-1, 1)
```

## Feature Engineering

### Statistical Features (Single-Channel)

| Feature | Formula | EMG Significance |
|---------|---------|------------------|
| **MAV** | MAV = (1/N)Σ\|xᵢ\| | Overall muscle activation |
| **ZCR** | ZCR = Σ𝕀(xᵢ·xᵢ₋₁ < 0) | Frequency proxy (critical for onset detection) |
| **SD** | SD = √[(1/N)Σ(xᵢ - x̄)²] | Signal energy |
| **MAX** | MAX = max(\|x\|) | Peak amplitude |

```python
def extract_statistical_features(window: np.ndarray) -> np.ndarray:
    """
    Extract time-domain features from single-channel window.

    Args:
        window: Shape (window_size, 1) or (window_size,) - single channel

    Returns:
        features: Shape (4,) - 4 statistical features
    """
    signal = window.flatten()

    mav = np.mean(np.abs(signal))
    zcr = np.sum(np.diff(np.sign(signal)) != 0)
    sd = np.std(signal)
    max_amp = np.max(np.abs(signal))

    return np.array([mav, zcr, sd, max_amp])


def extract_features_batch(X: np.ndarray) -> np.ndarray:
    """
    Extract features from all windows.

    Args:
        X: Shape (N, window_size, 1)

    Returns:
        X_features: Shape (N, 4)
    """
    return np.array([extract_statistical_features(w) for w in X])
```

### Temporal Features (Onset Detection)

For single-channel operation, temporal features become critical for word discrimination:

```python
def extract_temporal_features(window: np.ndarray, fs: int = 1000) -> np.ndarray:
    """
    Extract temporal features for onset/offset detection.

    Args:
        window: Preprocessed signal, shape (window_size,)
        fs: Sampling frequency

    Returns:
        features: Shape (4,) - temporal features
    """
    signal = window.flatten()
    n_samples = len(signal)

    # Divide into quarters
    q1 = signal[:n_samples//4]
    q2 = signal[n_samples//4:n_samples//2]
    q3 = signal[n_samples//2:3*n_samples//4]
    q4 = signal[3*n_samples//4:]

    # Energy in each quarter (captures temporal dynamics)
    e1 = np.mean(np.abs(q1))
    e2 = np.mean(np.abs(q2))
    e3 = np.mean(np.abs(q3))
    e4 = np.mean(np.abs(q4))

    return np.array([e1, e2, e3, e4])
```

## Exploratory Data Analysis

### Class Distribution

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(y: np.ndarray, label_map: dict):
    """Visualize class balance."""
    class_names = [label_map[i] for i in sorted(label_map.keys())]

    plt.figure(figsize=(10, 6))
    unique, counts = np.unique(y, return_counts=True)
    plt.bar([label_map[u] for u in unique], counts, color='steelblue')
    plt.title('Class Distribution')
    plt.xlabel('Word Class')
    plt.ylabel('Number of Windows')
    plt.savefig('eda_class_distribution.png', dpi=150)
    plt.show()
```

> **[INSERT IMAGE]** `images/eda_class_distribution.png`
> *Caption: Bar chart showing number of windows per word class (GHOST, LEFT, REST, STOP).*

```python
```

### Motor Intensity Comparison

```python
def plot_motor_intensity_comparison(data: dict):
    """
    Compare signal amplitudes across motor intensity levels.
    Visualizes the L3→L4 amplitude drop critical for transfer learning.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    levels = ['overt', 'whisper', 'mouthing', 'subvocal', 'imagined']
    titles = ['L1: Overt', 'L2: Whisper', 'L3: Mouthing', 'L4: Subvocal', 'L5: Imagined']

    for ax, level, title in zip(axes, levels, titles):
        if level in data:
            sample = data[level]['RawValue'].values[:3000]  # 3-second sample
            ax.plot(sample, linewidth=0.5)
            ax.set_title(title)
            ax.set_xlabel('Samples')

    axes[0].set_ylabel('ADC Value')
    plt.tight_layout()
    plt.savefig('eda_motor_intensity_comparison.png', dpi=150)
    plt.show()
```

> **[INSERT IMAGE]** `images/eda_motor_intensity_comparison.png`
> *Caption: Side-by-side comparison of signal amplitudes across all five motor intensity levels (L1-L5).*

### Descriptive Statistics

| Metric | Expected (L3 Mouthing) | Expected (L4 Subvocal) |
|--------|------------------------|------------------------|
| Mean MAV | Higher (~100-500) | Lower (~20-100) |
| Mean ZCR | Consistent | Similar (key insight!) |
| Amplitude Range | Wide | Narrow |

### Feature Space Visualization

```python
def plot_feature_scatter(X_features: np.ndarray, y: np.ndarray, label_map: dict):
    """
    2D scatter plot of MAV vs ZCR colored by class.
    """
    plt.figure(figsize=(10, 8))
    for class_id in sorted(label_map.keys()):
        mask = y == class_id
        plt.scatter(
            X_features[mask, 0],  # MAV
            X_features[mask, 1],  # ZCR
            label=label_map[class_id],
            alpha=0.6
        )
    plt.xlabel('Mean Absolute Value (MAV)')
    plt.ylabel('Zero Crossing Rate (ZCR)')
    plt.title('Feature Space Separability')
    plt.legend()
    plt.savefig('eda_feature_scatter.png', dpi=150)
    plt.show()
```

> **[INSERT IMAGE]** `images/eda_feature_scatter.png`
> *Caption: 2D scatter plot showing MAV vs ZCR colored by word class. Clusters indicate feature separability.*

## Key Observations

### From Phase 3 (Forearm EMG)
- **CLENCH** class forms distinct high-MAV, moderate-ZCR cluster
- **NOISE** class spans wide variance (requires non-linear decision boundaries)
- **Spectrograms** reveal frequency-specific textures useful for CNNs

### Expected for Phase 4 (Silent Articulation)

| Observation | Implication |
|-------------|-------------|
| **Lower amplitudes** | L4 signals ~10x smaller than L3; normalization critical |
| **ZCR stability** | Zero-crossing rate remains consistent across intensity levels |
| **Temporal patterns** | Word onset/offset timing is primary discriminator |
| **Confounds** | Swallowing artifacts may contaminate REST class |

### Single-Channel Mitigation Strategy

Without spatial discrimination (chin vs jaw ratio), the model must rely on:
1. **Temporal features** (~60% weight) — When activation occurs in the window
2. **Frequency features** (~30% weight) — ZCR, spectral characteristics
3. **Amplitude features** (~10% weight) — MAV, but less reliable

> **Key Insight:** The ZCR feature becomes especially critical for single-channel operation, as it captures frequency content independent of amplitude—enabling transfer from high-amplitude L3 training to low-amplitude L4 testing.
