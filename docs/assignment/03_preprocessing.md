# Section 3: Preprocessing, Cleaning & EDA

## Overview

Raw sEMG signals require preprocessing to remove noise and extract meaningful features. This section covers the signal processing pipeline and exploratory data analysis.

## Signal Processing Pipeline

```
Raw ADC → Bandpass 1-45Hz → Notch 60Hz → Normalization → Epoch → Features
```

## Preprocessing Steps

### 1. Bandpass Filtering

The AD8232's hardware filter (0.5-40Hz) is "accidentally perfect" for speech EMG (target: 1.3-50Hz per AlterEgo). We apply additional software filtering for consistency:

```python
from scipy.signal import butter, filtfilt, iirnotch

def bandpass_filter(signal: np.ndarray,
                    fs: int = 1000,
                    lowcut: float = 1.0,
                    highcut: float = 45.0,
                    order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.

    Args:
        signal: Raw EMG signal
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
        freq: Notch frequency (60Hz for US/Taiwan, 50Hz for EU)
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
```

## Feature Engineering

### Statistical Features (Per Channel)

| Feature | Formula | EMG Significance |
|---------|---------|------------------|
| **MAV** | MAV = (1/N)Σ\|xᵢ\| | Overall muscle activation |
| **ZCR** | ZCR = Σ𝕀(xᵢ·xᵢ₋₁ < 0) | Frequency proxy |
| **SD** | SD = √[(1/N)Σ(xᵢ - x̄)²] | Signal energy |
| **MAX** | MAX = max(\|x\|) | Peak amplitude |

```python
def extract_statistical_features(window: np.ndarray) -> np.ndarray:
    """
    Extract time-domain features from dual-channel window.

    Args:
        window: Shape (1000, 2) - dual channel window

    Returns:
        features: Shape (8,) - 4 features × 2 channels
    """
    features = []
    for ch in range(window.shape[1]):
        signal = window[:, ch]
        mav = np.mean(np.abs(signal))
        zcr = np.sum(np.diff(np.sign(signal)) != 0)
        sd = np.std(signal)
        max_amp = np.max(np.abs(signal))
        features.extend([mav, zcr, sd, max_amp])
    return np.array(features)
```

## Exploratory Data Analysis

### Class Distribution

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(y: np.ndarray, class_names: list):
    """Visualize class balance."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.xticks(range(len(class_names)), class_names)
    plt.title('Class Distribution')
    plt.xlabel('Word Class')
    plt.ylabel('Number of Windows')
    plt.savefig('eda_class_distribution.png')
    plt.show()
```

### Descriptive Statistics

| Metric | Channel 1 (Chin) | Channel 2 (Jaw) |
|--------|------------------|-----------------|
| Mean MAV | TBD | TBD |
| Mean ZCR | TBD | TBD |
| Std Dev | TBD | TBD |

### Feature Space Visualization

```python
def plot_feature_scatter(X_features: np.ndarray, y: np.ndarray):
    """
    2D scatter plot of MAV vs ZCR colored by class.
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_features[:, 0],  # MAV Channel 1
        X_features[:, 1],  # ZCR Channel 1
        c=y,
        cmap='viridis',
        alpha=0.6
    )
    plt.colorbar(scatter, label='Class')
    plt.xlabel('Mean Absolute Value (Ch1)')
    plt.ylabel('Zero Crossing Rate (Ch1)')
    plt.title('Feature Space Separability')
    plt.savefig('eda_feature_scatter.png')
    plt.show()
```

## Key Observations (Phase 3 Insights)

From the Phase 3 sEMG study:
- **CLENCH** class forms distinct high-MAV, moderate-ZCR cluster
- **NOISE** class spans wide variance (needs non-linear boundaries)
- **Spectrograms** reveal frequency-specific textures

For Phase 4 (subvocalization), we expect:
- Lower overall signal amplitude (covert vs overt speech)
- More subtle inter-word differences
- Dual-channel providing complementary information
