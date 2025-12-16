# Section 2: Data Loading & Python Conversion

## Overview

This section covers converting raw dual-channel EMG signals from the ESP32 serial stream into Python-readable formats compatible with scikit-learn and deep learning frameworks.

## Data Acquisition Pipeline

```
ESP32 Serial → CSV Files → Pandas DataFrame → NumPy Arrays
```

## Hardware Interface

### ESP32 Firmware Specifications
- **ADC Resolution:** 12-bit (0-4095 range)
- **Sampling Rate:** 1000Hz per channel
- **Output Format:** Serial @ 115200 baud
- **Channels:** 2 (chin + jaw)

## Code: Data Loading

```python
import pandas as pd
import numpy as np
from glob import glob
import os

def load_session_data(session_dir: str) -> pd.DataFrame:
    """
    Load a single recording session from CSV files.

    Args:
        session_dir: Path to session directory containing CSV files

    Returns:
        DataFrame with columns: timestamp, ch1_voltage, ch2_voltage, label
    """
    csv_files = sorted(glob(os.path.join(session_dir, "*.csv")))
    dfs = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, names=['timestamp', 'ch1', 'ch2', 'label'])
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_all_sessions(data_dir: str) -> pd.DataFrame:
    """
    Load all recording sessions from the data directory.

    Args:
        data_dir: Root data directory containing session subdirectories

    Returns:
        Combined DataFrame with session_id column added
    """
    session_dirs = sorted(glob(os.path.join(data_dir, "Session*")))
    all_data = []

    for i, session_dir in enumerate(session_dirs, 1):
        df = load_session_data(session_dir)
        df['session_id'] = i
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)
```

## Code: Windowing for ML Pipeline

```python
def create_windows(df: pd.DataFrame,
                   window_size: int = 1000,
                   overlap: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment continuous stream into fixed-size windows.

    Args:
        df: Raw data DataFrame
        window_size: Samples per window (1000 = 1 second @ 1000Hz)
        overlap: Fraction of overlap between windows (0.0 = non-overlapping)

    Returns:
        X: Array of shape (n_windows, window_size, 2) - dual channel
        y: Array of shape (n_windows,) - labels
    """
    step = int(window_size * (1 - overlap))
    windows_X = []
    windows_y = []

    ch1 = df['ch1'].values
    ch2 = df['ch2'].values
    labels = df['label'].values

    for start in range(0, len(df) - window_size, step):
        end = start + window_size
        window_ch1 = ch1[start:end]
        window_ch2 = ch2[start:end]

        # Stack channels: (window_size, 2)
        window = np.stack([window_ch1, window_ch2], axis=-1)
        windows_X.append(window)

        # Majority vote for window label
        label = np.median(labels[start:end]).astype(int)
        windows_y.append(label)

    return np.array(windows_X), np.array(windows_y)
```

## Data Structure Summary

| Structure | Shape | Description |
|-----------|-------|-------------|
| `X_raw` | `(N, 1000, 2)` | Raw dual-channel windows |
| `y` | `(N,)` | Class labels (0=REST, 1=GHOST, 2=LEFT, ...) |
| `X_features` | `(N, 8)` | Statistical features (4 per channel) |
| `X_spectrograms` | `(N, 224, 224, 3)` | Mel-spectrogram images |

## Dependencies

```python
# requirements.txt
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
pyserial>=3.5  # For real-time acquisition
```
