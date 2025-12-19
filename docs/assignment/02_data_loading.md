# Section 2: Data Loading & Python Conversion

## Overview

This section covers converting raw single-channel EMG signals from the ESP32 serial stream into Python-readable formats compatible with scikit-learn and deep learning frameworks.

## Data Acquisition Pipeline

```
ESP32 Serial → CSV Files → Pandas DataFrame → NumPy Arrays
```

## Hardware Interface

### ESP32 Firmware Specifications
- **ADC Resolution:** 12-bit (0-4095 range)
- **Sampling Rate:** 1000Hz
- **Baud Rate:** 230400
- **Output Format:** `Timestamp,RawValue` per line
- **Channels:** 1 (single AD8232, under-chin placement)

## CSV File Format

The capture tools produce CSV files with the following structure:

```csv
Label,Timestamp,RawValue
GHOST,1234567,2048
GHOST,1234568,2052
LEFT,1234569,2045
...
```

| Column | Type | Description |
|--------|------|-------------|
| `Label` | string | Word class (GHOST, LEFT, STOP, REST) |
| `Timestamp` | int | Millisecond timestamp from ESP32 `millis()` |
| `RawValue` | int | ADC reading (0-4095) |

## Code: Data Loading

```python
import pandas as pd
import numpy as np
from glob import glob
import os

def load_spectrum_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load all motor intensity spectrum CSV files.

    Args:
        data_dir: Path to speech-capture directory

    Returns:
        Dictionary mapping level names to DataFrames
    """
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
            print(f"Loaded {level}: {len(df)} samples")

    return data


def load_single_file(filepath: str) -> pd.DataFrame:
    """
    Load a single CSV recording file.

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with columns: Label, Timestamp, RawValue
    """
    return pd.read_csv(filepath)
```

## Code: Windowing for ML Pipeline

```python
def create_windows(df: pd.DataFrame,
                   window_size: int = 3000,
                   overlap: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment continuous stream into fixed-size windows.

    Args:
        df: Raw data DataFrame with Label, Timestamp, RawValue columns
        window_size: Samples per window (3000 = 3 seconds @ 1000Hz)
        overlap: Fraction of overlap between windows (0.0 = non-overlapping)

    Returns:
        X: Array of shape (n_windows, window_size, 1) - single channel
        y: Array of shape (n_windows,) - string labels
    """
    step = int(window_size * (1 - overlap))
    windows_X = []
    windows_y = []

    values = df['RawValue'].values
    labels = df['Label'].values

    for start in range(0, len(df) - window_size, step):
        end = start + window_size
        window = values[start:end].reshape(-1, 1)  # (window_size, 1)
        windows_X.append(window)

        # Majority vote for window label
        window_labels = labels[start:end]
        label = pd.Series(window_labels).mode()[0]
        windows_y.append(label)

    return np.array(windows_X), np.array(windows_y)


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Convert string labels to integer encoding.

    Args:
        y: Array of string labels

    Returns:
        y_encoded: Integer-encoded labels
        label_map: Dictionary mapping integers to strings
    """
    unique_labels = sorted(set(y))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}

    y_encoded = np.array([label_to_int[label] for label in y])
    return y_encoded, int_to_label
```

## Code: Train/Test Split by Motor Intensity

```python
def prepare_transfer_learning_split(data: dict) -> tuple:
    """
    Prepare train/test split following the transfer learning protocol.

    Training: Mouthing (L3) - high SNR
    Testing: Subvocal (L4) - low SNR

    Args:
        data: Dictionary from load_spectrum_data()

    Returns:
        X_train, y_train, X_test, y_test
    """
    # Training data: Mouthing (Level 3)
    X_train, y_train = create_windows(data['mouthing'])

    # Testing data: Subvocal (Level 4)
    X_test, y_test = create_windows(data['subvocal'])

    # Encode labels
    y_train, label_map = encode_labels(y_train)
    y_test, _ = encode_labels(y_test)

    print(f"Training: {len(X_train)} windows from Mouthing")
    print(f"Testing: {len(X_test)} windows from Subvocal")
    print(f"Label map: {label_map}")

    return X_train, y_train, X_test, y_test, label_map
```

## Data Structure Summary

| Structure | Shape | Description |
|-----------|-------|-------------|
| `X_raw` | `(N, 3000, 1)` | Raw single-channel windows (3s @ 1000Hz) |
| `y` | `(N,)` | Class labels (0=GHOST, 1=LEFT, 2=REST, 3=STOP) |
| `X_features` | `(N, 4)` | Statistical features per window |
| `X_spectrograms` | `(N, 224, 224, 3)` | Mel-spectrogram images |

## Expected Data Files

```
phase4/speech-capture/
├── overt_data.csv      # L1: Speaking out loud (calibration)
├── whisper_data.csv    # L2: Whispering (calibration)
├── mouthing_data.csv   # L3: Open-mouth → TRAINING
├── subvocal_data.csv   # L4: Closed-mouth → TESTING
└── imagined_data.csv   # L5: Pure mental (exploratory)
```

### Actual Data Counts (from Colab)

| File | Samples | Duration | Windows |
|------|---------|----------|---------|
| `overt_data.csv` | 30,096 | ~30s | 10 |
| `whisper_data.csv` | 30,192 | ~30s | 10 |
| `mouthing_data.csv` | 515,547 | ~515s | **200** |
| `subvocal_data.csv` | 537,901 | ~538s | **201** |
| `imagined_data.csv` | 107,791 | ~108s | 36 |
| **Total** | **1,221,527** | **~20 min** | **457** |

### Window Creation Output

```
Creating windows (extracting MIDDLE 1-second where word was spoken)...
📊 Mouthing (L3 - Training): (200, 1000, 1)
📊 Subvocal (L4 - Testing): (201, 1000, 1)
```

> **Key Detail:** Windows are 1-second (1000 samples) extracted from the MIDDLE of each 3-second block (samples 1000-2000), capturing the moment when the word was vocalized at countdown "2".

## Dependencies

```python
# requirements.txt
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
pyserial>=3.5  # For real-time acquisition
```
