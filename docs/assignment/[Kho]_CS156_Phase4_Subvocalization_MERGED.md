# CS156 Pipeline - Final Draft
## Phase 4: Subvocalization Detection with Low-Cost Hardware

**Student:** Carl Vincent Kho
**Generated:** 2025-12-20 02:54:56
**Course:** CS156 - Machine Learning Pipeline

---

## Table of Contents

1. [Section 1: Data Explanation](#section-1-data-explanation)
2. [Section 2: Data Loading & Python Conversion](#section-2-data-loading-and-python-conversion)
3. [Section 3: Preprocessing, Cleaning & EDA](#section-3-preprocessing,-cleaning-and-eda)
4. [Section 4: Analysis Plan & Data Splits](#section-4-analysis-plan-and-data-splits)
5. [Section 5: Model Selection & Mathematical Foundations](#section-5-model-selection-and-mathematical-foundations)
6. [Section 6: Model Training](#section-6-model-training)
7. [Section 7: Predictions & Performance Metrics](#section-7-predictions-and-performance-metrics)
8. [Section 8: Visualization & Conclusions](#section-8-visualization-and-conclusions)
9. [Section 9: Executive Summary](#section-9-executive-summary)
10. [Section 10: References](#section-10-references)

---



<div style='page-break-before: always;'></div>

# Section 1: Data Explanation

## Dataset Overview

This project uses **single-channel surface EMG (sEMG) signals** captured from submental muscles during silent speech tasks. The data represents an extension of the Phase 3 single-lead forearm EMG dataset to silent speech recognition. An attempt to replicate MIT Media Lab's **AlterEgo** system (Kapur et al., 2018) for **$30** instead of **$1,200+**.

> **Note on Hardware Adaptation:** This project was originally designed as a dual-channel system (chin + jaw). Due to hardware limitations discovered during testing—one AD8232 exhibited ADC saturation near the 12-bit ceiling and Carl flew away from his electronics supplier. Thus, the system was adapted to single-channel operation. Full troubleshooting documentation is available in the [working_process/](https://github.com/CarlKho-Minerva/CP-PHASE4_Subvocalization_25TPE/tree/main/docs/working_process) directory.

## Data Source

### Personal Digital Archive Origin
- **Creator:** Carl Vincent Ladres Kho (Minerva University)
- **Collection Period:** December 2025
- **Location:** Taipei, Taiwan (components from Guang Hua Digital Plaza)
- **Context:** Final assignment for CS156 Machine Learning Pipeline

---

## Hardware Acquisition: The Guang Hua Run

Components were purchased from **Jin Hua Electronics (今華電子)** in Guang Hua Digital Plaza, Taipei.

### Component List (~$30 USD Total)

| Component | Source | Cost (TWD) | Purpose |
|-----------|--------|------------|---------|
| **AD8232 x2** | Jin Hua Electronics | ~$300 each for a set (includes gold-plated jack and three Ag/AgCl Electrodes) | Originally dual-channel; one unit exhibited saturation |
| **ESP32 (NodeMCU-32S)** | Jin Hua | ~$180 | MCU @ 1000Hz sampling, 3.3V logic |
| **Ag/AgCl Electrodes (5-pack)** | Medical supply | $40 | Conductive gel pads with metal snap |
| **USB Power Bank** | Existing (Macbook) | - | **Safety: NEVER use wall power** |

![alt text](image.png)
> *Caption: Full hardware stack including AD8232 sensors, ESP32, and custom cables.*

### The "Accidental Hardware Match"

The **AD8232** is designed for ECG (heart monitoring), but its hardware bandpass filter is accidentally perfect for speech EMG:

| System | Target Frequency | AD8232 Filter |
|--------|------------------|---------------|
| MIT AlterEgo | 1.3Hz – 50Hz | ✓ |
| AD8232 Native | 0.5Hz – 40Hz | Hardware match! |

---

## The Problem

### Original Design Intent

The initial design followed AlterEgo's multi-site electrode approach:
- **Channel 1 (Digastric/Mylohyoid):** Under-chin placement for tongue position tracking
- **Channel 2 (Masseter):** Jaw/cheek placement for bite intensity measurement

### Hardware Limitation Discovered

During validation testing (December 19, 2025), the two AD8232 units exhibited significantly different baseline ADC characteristics:

| Sensor | Baseline ADC | Heart LED | Operational Status |
|--------|--------------|-----------|-------------------|
| AD8232 #1 (Red PCB) | ~1,800 | Flickering | Functional |
| AD8232 #2 (Purple PCB) | ~3,800 | Not flickering | Saturation risk |

The second sensor's baseline near the 12-bit ADC ceiling (4095) meant that any muscle activation would saturate the signal, resulting in clipped waveforms and loss of amplitude information. Serial monitor output during testing:

```
ADC: 3796 | LO+: 1 | LO-: 1 | Status: ✓ Board responding
ADC: 3823 | LO+: 1 | LO-: 1 | Status: ✓ Board responding
ADC: 3921 | LO+: 1 | LO-: 1 | Status: ⚠️ ADC RAILING HIGH
```

<div>
    <a href="https://www.loom.com/share/a893fc0e55334356979a57ffecdbcfa3">
      <p>PHASE 4 SOMACH - 2nd AD8232 Broken (there's hope) - Watch Video</p>
    </a>
    <a href="https://www.loom.com/share/a893fc0e55334356979a57ffecdbcfa3">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/a893fc0e55334356979a57ffecdbcfa3-c7eead79137547ac-full-play.gif#t=0.1">
    </a>
  </div>

> *Caption: Video documentation of the troubleshooting session identifying the saturation issue.*

### The Pivot

Given the hardware constraint, a decision was made to proceed with single-channel data collection using the functional AD8232 unit. This decision was informed by analysis of feature discrimination capabilities:

| Feature Type | Dual-Channel | Single-Channel | Notes |
|--------------|--------------|----------------|-------|
| Spatial (chin vs jaw ratio) | Available | Lost | Cannot compare channel ratios |
| Temporal (firing sequence) | Available | Preserved | Primary discriminator |
| Frequency (ZCR, spectral) | Available | Preserved | Secondary discriminator |
| Amplitude (signal strength) | Available | Reduced | Lower confidence without reference |

**Mitigation Strategy:** With single-channel operation, the classification model must rely primarily on **temporal features** (onset timing, duration, activation sequence) and **frequency features** (zero-crossing rate, spectral characteristics) rather than spatial discrimination between electrode sites.

## Electrode Placement

### Anatomical Targets

The electrode placement targets the **Digastric/Mylohyoid muscle group**, which controls tongue elevation and depression during speech articulation.

![Digastric muscle anatomy](images/anatomy_digastric_muscle.png)
*Figure 1: Lateral view of neck musculature with digastric muscle highlighted (red). This muscle group contracts during tongue movement, producing the sEMG signal detected by under-chin electrodes. (Source: Gray's Anatomy, Public Domain, via Wikimedia Commons)*

The ground electrode is placed on the **Mastoid Process**—the bony protrusion behind the ear—which provides an electrically neutral reference point away from active musculature.

![Mastoid process anatomy](images/anatomy_mastoid_process.png)
*Figure 2: Skull showing mastoid process (red) of the temporal bone. This location is ideal for ground electrode placement due to minimal muscle activity and proximity to the signal electrodes. (Source: Wikimedia Commons, CC BY-SA 4.0)*

### Single-Channel Configuration

![Electrode placement schematic showing chin positions](images/viz_electrode_schematic.png)

**Digastric/Mylohyoid (Under-Chin) — Tongue Tracker**

| Electrode | Placement | Purpose |
|-----------|-----------|---------|
| **Signal+ (Red)** | Under-chin, left of centerline | Tongue movement detection |
| **Signal- (Yellow)** | Under-chin, right of centerline, 2-3cm apart | Differential signal |
| **Reference (Green)** | Mastoid process (behind ear) | Electrically neutral ground |

![alt text](image-1.png)
> *Caption: Electrode placement under the chin targeting the Digastric muscle.*

### 3.5mm Jack Wiring Mapping

Verified experimentally (see [2025-12-18_wiring_mapping_session.md](../working_process/2025-12-18_wiring_mapping_session.md)):

| 3.5mm Plug Segment | Wire Color | Body Placement |
|--------------------|------------|----------------|
| **Tip** | Yellow | Signal- (right of centerline) |
| **Ring (Middle)** | Green | Reference (mastoid) |
| **Sleeve (Back)** | Red | Signal+ (left of centerline) |

---

## Hardware Validation Protocol

To ensure signal integrity before data collection, a 3-step "Parking Lot Test" was performed:

**Step 1: Heartbeat Sanity Check**
- Electrodes on chest.
- **Success Criteria:** Clean, rhythmic QRS complex (heartbeat) every ~1s.
- **Purpose:** Verifies sensor and ADC functionality.

**Step 2: Jaw Clench Noise Check**
- Electrodes on Jaw. Bite down hard.
- **Success Criteria:** Signal "explodes" into high-amplitude chaos (>2000 units).
- **Purpose:** Verifies electrodes are making contact and amplifier isn't saturated.

**Step 3: Subvocalization "Wiggle"**
- Electrodes on Chin. Say "GHOST" internally.
- **Success Criteria:** Small but distinct disturbance from baseline noise.
- **Purpose:** Confirms detection of fine motor units in the tongue.

---

## Vocabulary Selection

Words were chosen based on **distinct neuromuscular signatures**, not semantic meaning.

### The Insight

> *"You are building a Biological Keyboard, not a Telepathy Helmet."*

Since electrodes are under the chin, I'm tracking **tongue position**, not sound. Choose words that force the tongue to do radically different things.

### Tier 1: High Success Rate

| Word | Tongue Physics | Expected Signal |
|------|----------------|-----------------|
| **GHOST** | Back of tongue → soft palate ("G" slam) | High-frequency burst |
| **LEFT** | Tongue tip → alveolar ridge ("L" touch) | Distinct onset |
| **STOP** | Plosive "ST" + "P" = jaw engagement | Combined signal |
| **REST** | Tongue flat, relaxed | Control (silence) |

### Tier 2: Control Word

**"MAMA"** — Lips only (Orbicularis Oris). Tongue stays flat.
- **Purpose:** If you subvocalize "MAMA" and see a chin signal spike, you're picking up **noise**, not muscle.

## The Motor Intensity Spectrum

### 5-Level "Descending Motor Intensity" Framework

To validate the low-cost hardware, I employ a **Transfer Learning** strategy across the motor intensity spectrum.

> **The Insight:** Training on "Open Mouth" movements (Mouthing) provides strong, high-amplitude signals that help the model learn the temporal dynamics of each word. I then transfer this knowledge to "Closed Mouth" (Silent Articulation) scenarios.

| Level | Terminology | Description | Signal | Role |
|-------|-------------|-------------|--------|------|
| 1 | **Overt Speech** | Natural speaking voice | 🔊🔊🔊🔊🔊 | Calibration |
| 2 | **Whisper** | Low-volume vocalization | 🔊🔊🔊🔊 | Calibration |
| 3 | **Mouthing** | **Open-Mouth** silent speech with maximal jaw excursion | 🔊🔊🔊 | **Training Data** (Source) |
| 4 | **Silent Articulation** | **Closed-Mouth** speech with exaggerated internal tongue movement | 🔊🔊 | **Testing Data** (Target) |
| 5 | **Imagined Speech** | Minimal/Micro-movements (Reading to self) | 🔊 | Exploratory |

### Data Collection Summary

Data was collected across all five motor intensity levels on December 19, 2025.

| Level | Cycles | Total Samples | Output File |
|-------|--------|---------------|-------------|
| L1 Overt | 10 | 30,096 | `overt_data.csv` |
| L2 Whisper | 10 | 30,192 | `whisper_data.csv` |
| L3 Mouthing | 50 | 515,547 | `mouthing_data.csv` |
| L4 Subvocal | 51 | 537,901 | `subvocal_data.csv` |
| L5 Imagined | 10 | 107,791 | `imagined_data.csv` |
| **Total** | **131** | **1,221,527** | - |

#### Class Balance (Verified in Colab)
```
MOUTHING:  GHOST=129,354 (25.1%) | STOP=128,853 (25.0%) | LEFT=128,703 (25.0%) | REST=128,591 (24.9%)
SUBVOCAL:  GHOST=135,906 (25.3%) | REST=134,032 (24.9%) | STOP=133,993 (24.9%) | LEFT=133,970 (24.9%)
```

#### Signal Statistics (Raw ADC Values)

| Level | Mean | Std | Min | Max | Range |
|-------|------|-----|-----|-----|-------|
| OVERT | 1921.37 | 12.31 | 1 | 1989 | 1988 |
| WHISPER | 1921.02 | 8.98 | 1857 | 1987 | 130 |
| MOUTHING | 1921.18 | 9.75 | 1853 | 1991 | 138 |
| SUBVOCAL | 1921.15 | 260.62 | 22 | 192921* | 192899 |
| IMAGINED | 1921.28 | 9.55 | 1858 | 1987 | 129 |

> ** Data Anomaly:** Subvocal data contained 1 outlier sample with value 192,921 (likely a sensor glitch). This was removed during preprocessing with a `RawValue < 4000` filter.

#### Per-Class Statistics (Mouthing)

| Class | Mean | Std | Range |
|-------|------|-----|-------|
| GHOST | 1921.2 | 9.7 | [1855, 1987] |
| LEFT | 1921.1 | 9.7 | [1853, 1989] |
| STOP | 1921.2 | 9.8 | [1854, 1991] |
| REST | 1921.2 | 9.8 | [1856, 1989] |

> **SMOKING GUN:** All four word classes have **identical** mean (1921.2) and standard deviation (9.7-9.8). This indicates that the single-channel signal contains **no discriminative information** for word-level classification. The signal can detect *that* muscle activation occurred, but cannot distinguish *which word* was articulated.

### Transfer Learning Rationale
**Open (Level 3) → Closed (Level 4)**

I assume that the *temporal sequence* of muscle activation (e.g., G-H-O-S-T) remains consistent between open and closed mouth states, even if the *amplitude* changes.
- **Training (Level 3):** Learn the neuromuscular "signature" of the word with high Signal-to-Noise Ratio (SNR).
- **Inference (Level 4):** Detect the same signature in the constrained, closed-mouth environment.

---

## Known Confounds and Limitations

The following artifacts were observed during data collection and documented for transparency:

| Confound | Description | Potential Impact |
|----------|-------------|------------------|
| **Saliva Swallowing** | Periodic swallowing creates spurious EMG bursts unrelated to speech | May contaminate REST class labels |
| **Post-Mouthing Muscle Tension** | After extensive L3 trials, chin muscles remain partially activated | REST labels after mouthing may not represent true baseline |
| **Syllabic Beat Artifact** | Involuntary micro-movements synchronized to word rhythm, even during L5 | "Imagined speech" may contain detectable motor artifacts |
| **Recording Timing Convention** | Words vocalized at countdown "2" to center signal in analysis window | Consistent across all levels; enables time-alignment |

---

## Prior Work Context

This dataset builds on **Phase 3** (Kho, 2025), which validated:
- AD8232 sensor efficacy for EMG capture (SDN pin fix discovered)
- 18 ML architecture benchmark on 1.54M data points
- Random Forest as Pareto-optimal for ESP32 deployment (74% accuracy, 0.01ms)
- MaxCRNN achieving 99% precision on safety-critical class

---

## Sampling Methodology

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Sampling Rate** | 1000Hz | Nyquist: fₛ > 2×450Hz (EMG bandwidth) |
| **ADC Resolution** | 12-bit (0-4095) | ESP32 native |
| **Window Size** | 3 seconds per word | Countdown-aligned capture |
| **Channels** | 1 (single AD8232) | Under-chin only (hardware constraint) |
| **Power Source** | USB Battery Bank | **Safety: No wall power with face electrodes** |

---

## References

1. Kapur, A., Kapur, S., & Maes, P. (2018). AlterEgo: A personalized wearable silent speech interface. *Proceedings of the 23rd International Conference on Intelligent User Interfaces*, 43-53. https://doi.org/10.1145/3172944.3172977

2. Kho, C. V. (2025). Phase 3: EMG-Based Gesture Classification with AD8232. *Minerva University CS156 Project Archive*.

3. Analog Devices. (2012). AD8232: Single-Lead, Heart Rate Monitor Front End. *Datasheet*. https://www.analog.com/media/en/technical-documentation/data-sheets/ad8232.pdf

4. Gray, H. (1918). *Anatomy of the Human Body*. Philadelphia: Lea & Febiger. (Digastric muscle illustration, Public Domain)


---



<div style='page-break-before: always;'></div>

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


---



<div style='page-break-before: always;'></div>

# Section 3: Preprocessing, Cleaning & EDA

## Overview

Raw sEMG signals require preprocessing to remove noise and extract meaningful features. This section covers the signal processing pipeline and exploratory data analysis for single-channel subvocalization data.

## Signal Processing Pipeline

```
Raw ADC → Bandpass 1-45Hz → Notch 60Hz → Normalization → Epoch → Features
```

> **Note:** Again, the AD8232's hardware bandpass filter (0.5-40Hz) already provides substantial filtering aligned with AlterEgo's target range (1.3-50Hz). Software filters are applied for consistency and to remove residual power line interference.

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

## Actual Observations (from Colab)

### Per-Class Statistics (Mouthing)

```
GHOST: mean=1921.2, std=9.7, range=[1855, 1987]
LEFT:  mean=1921.1, std=9.7, range=[1853, 1989]
STOP:  mean=1921.2, std=9.8, range=[1854, 1991]
REST:  mean=1921.2, std=9.8, range=[1856, 1989]
```

> **SMOKING GUN:** All four word classes have **identical** statistics. Mean, standard deviation, and range are indistinguishable between GHOST, LEFT, STOP, and REST.

### Signal Statistics Across Motor Intensity Levels

| Level | Mean | Std | Range |
|-------|------|-----|-------|
| OVERT | 1921.37 | 12.31 | 1988 |
| WHISPER | 1921.02 | 8.98 | 130 |
| MOUTHING | 1921.18 | 9.75 | 138 |
| SUBVOCAL | 1921.15 | 260.62* | 192899* |
| IMAGINED | 1921.28 | 9.55 | 129 |

*Subvocal contained 1 outlier (192,921) removed during preprocessing.

### Block Length Statistics

| Level | Mean (samples) | Mean (seconds) | Total Blocks |
|-------|----------------|----------------|--------------|
| MOUTHING | 2,578 | 2.58s | 200 |
| SUBVOCAL | 2,676 | 2.68s | 201 |

### Why Preprocessing Couldn't Help

| Feature Type | Phase 3 (Forearm) | Phase 4 (Subvocal) |
|--------------|-------------------|-------------------|
| Amplitude (MAV) | Distinct per class | **Identical** (1921.2) |
| Frequency (ZCR) | Stable, discriminative | Identical between words |
| Temporal (onset) | Clear patterns | No visible patterns |

> **Conclusion:** Preprocessing and feature engineering cannot extract discriminative information that doesn't exist in the raw signal. The single-channel setup captures muscle activation but cannot distinguish between different tongue positions.

### Visualization Evidence

![Random samples mouthing](images/viz_random_samples_mouthing.png)
*Random samples per class (Mouthing): All 4 word classes show visually indistinguishable waveforms.*

![ADC distribution](images/viz_adc_distribution.png)
*ADC distribution: Mouthing (broad) and Subvocal (narrow spike) both centered at 1921.*

![Spectrograms](images/viz_spectrograms.png)
*Mel-Spectrograms: All 4 classes show identical frequency content.*


---



<div style='page-break-before: always;'></div>

# Section 4: Analysis Plan & Data Splits

## Overview

This section describes the classification task and train/test split methodology for the single-channel subvocalization pipeline.

## Classification Task

### Problem Definition

**Multi-class classification** of subvocalized words from single-channel sEMG signals.

| Aspect | Description |
|--------|-------------|
| **Task Type** | Multi-class Classification |
| **Input** | Single-channel sEMG window (3000×1) |
| **Output** | Word class (GHOST, LEFT, STOP, REST) |
| **Metric** | Accuracy, Precision, F1-Score |

### Transfer Learning Strategy

The key insight of Phase 4 is **transfer learning from overt to covert speech**:

```
Level 3 (Mouthing)         → Train (High SNR, exaggerated movements)
         ↓
Level 4 (Silent Articulation) → Test (Low SNR, constrained movements)
```

> **Note:** Transfer learning from Mouthing to Silent Articulation is based on the assumption that the temporal sequence of muscle activation is consistent between Open and Closed states, even if the amplitude differs by an order of magnitude.

### Target Classes (4 Classes)

| Class | Word | Tongue Physics |
|-------|------|----------------|
| 0 | **GHOST** | Back of tongue → soft palate (velar stop) |
| 1 | **LEFT** | Tongue tip → alveolar ridge (lateral approximant) |
| 2 | **REST** | Tongue flat, relaxed (null class) |
| 3 | **STOP** | Plosive onset, jaw engagement |

> **Note:** "MAMA" is used only for hardware validation (lip movement = no tongue signal), not classification.

## Data Split Strategy

### Strategy: Transfer Learning across Motor Intensities

The core hypothesis is that models trained on **Mouthing (Open Articulation)** can generalize to **Silent Articulation (Closed Articulation)**.

> **Why this matters:** I assume the *temporal sequence* of muscle activation is consistent between Open and Closed states, even if the *amplitude* differs by an order of magnitude.

| Split | Source | Data | Rationale |
|-------|--------|------|-----------|
| **Train** | Level 3: Mouthing | ~50 cycles × 4 words | High-amplitude, exaggerated signals to learn temporal dynamics |
| **Validation** | Level 3: Mouthing (held out) | ~10% of L3 | Hyperparameter tuning on source domain |
| **Test** | Level 4: Silent Articulation | ~50 cycles × 4 words | Low-amplitude, constrained signals (real-world scenario) |

![ADC distribution](images/viz_adc_distribution.png)
*Figure: ADC value distribution showing Mouthing vs Subvocal.*

### Single-Channel Considerations

Without dual-channel spatial features, the model relies on:

| Feature Type | Importance | Notes |
|--------------|------------|-------|
| **Temporal patterns** | ***** | Primary discriminator (onset timing, duration) |
| **Frequency features** | **** | ZCR critical (stable across amplitude changes) |
| **Amplitude features** | ** | Less reliable for transfer (L3→L4 amplitude drop) |

### Implementation in Code

```python
def create_transfer_splits(data: dict) -> tuple:
    """
    Create train/test split for transfer learning.

    Args:
        data: Dictionary with 'mouthing' and 'subvocal' DataFrames

    Returns:
        X_train, y_train, X_test, y_test
    """
    from sklearn.model_selection import train_test_split

    # Source Domain: Level 3 (Mouthing)
    X_source, y_source = create_windows(data['mouthing'])

    # Target Domain: Level 4 (Silent Articulation)
    X_target, y_target = create_windows(data['subvocal'])

    # Train/Val split on source domain only
    X_train, X_val, y_train, y_val = train_test_split(
        X_source, y_source, test_size=0.15, random_state=42, stratify=y_source
    )

    # Test set is entirely target domain
    X_test, y_test = X_target, y_target

    return X_train, X_val, X_test, y_train, y_val, y_test
```

## Evaluation Metrics

| Metric | Description | Priority |
|--------|-------------|----------|
| **Accuracy** | Overall correctness across all 4 classes | Primary |
| **F1-Score (Macro)** | Balanced metric for class imbalance | Secondary |
| **Confusion Matrix** | Visualize specific misclassifications | Diagnostic |
| **Inference Latency** | Must be <5ms per window for real-time ESP32 | Deployment |

### Success Criteria vs. Actual Results

| Level | Target | Actual (L4 Test) | Status |
|-------|--------|------------------|--------|
| **Baseline** | >50% | 24.38% | Failed |
| **Acceptable** | >65% | 24.38% | Failed |
| **Target** | >80% | 24.38% | Failed |
| **Binary (WORD vs REST)** | >50% | 72.64% (mode collapse) | **Failed** |

> **Critical Finding:** All classification failed. Multi-class at chance level (25%), binary collapsed to majority-class prediction (model predicts WORD for 100% of REST samples).

### Comparison to Phase 3 Results

| Metric | Phase 3 (Forearm) | Phase 4 (Subvocal) |
|--------|-------------------|-------------------|
| Classes | 3 (RELAX, CLENCH, NOISE) | 4 (GHOST, LEFT, STOP, REST) |
| Channels | 1 | 1 |
| Best Model | Random Forest (74%) | None viable |
| Multi-class Accuracy | 74% | 24% (failed) |
| Target Signal | Flexor Digitorum (large) | Digastric (tiny) |
| SNR | High | Very Low |

### Why Phase 4 Failed (Root Cause Analysis)

| Factor | Phase 3 | Phase 4 | Impact |
|--------|---------|---------|--------|
| Muscle Size | Large forearm muscle | Tiny submental muscles | 10-100× weaker signal |
| Word Discrimination | N/A (gesture vs rest) | 4 distinct words | Per-class stats identical |
| Spatial Info | N/A (single site ok) | Lost (need 2+ sites) | Cannot distinguish tongue positions |

## Analysis Pipeline Steps

```
1. Data Loading    → Load 5 CSV files (L1-L5)
2. Preprocessing   → Bandpass, Notch 60Hz, Normalize
3. Windowing       → 3-second windows per word
4. Feature Extract → Statistical (MAV, ZCR, SD, MAX) + Temporal
5. Train/Val/Test  → L3→Train/Val, L4→Test
6. Model Training  → Random Forest baseline, then MaxCRNN
7. Evaluation      → Confusion matrix, F1-score on L4
8. Visualization   → Feature distributions, t-SNE embeddings
```

## Safety Considerations

Following Phase 3's findings, I prioritize:

| Constraint | Value | Rationale |
|------------|-------|-----------|
| **Precision** | >90% on active classes | Avoid false positives in control applications |
| **Latency** | <100ms end-to-end | Real-time feedback requirement |
| **Memory** | <320KB model size | ESP32 SRAM constraint |

The test set evaluation will focus on these deployment constraints.

## Exploratory Analysis: Multi-Level Validation

Beyond the primary L3→L4 transfer, I collect L1, L2, L5 data for exploratory analysis:

| Level | Purpose |
|-------|---------|
| L1 (Overt) | Calibration baseline; verify signal quality |
| L2 (Whisper) | Intermediate amplitude; validate fade curve |
| L5 (Imagined) | Future work; pure mental representation |



---



<div style='page-break-before: always;'></div>

# Section 5: Model Selection & Mathematical Foundations

## Overview

This section discusses model selection for **Silent Articulation classification** from single-channel sEMG signals, including a **novel technique not covered in class**: the **MaxCRNN** (Inception + Bi-LSTM + Attention) architecture.

## Model Selection Rationale

### The "Ladder of Abstraction"

Following Phase 3's methodology, I evaluate models across increasing complexity:

| Tier | Models | Feature Set | Compute |
|------|--------|-------------|---------|
| **Heuristics** | Threshold, Variance | Raw amplitude | O(N) |
| **Classical ML** | Random Forest, SVM | Statistical (MAV, ZCR, SD, MAX) | O(N log N) |
| **Deep Learning** | 1D CNN, CRNN | Raw sequence | O(N²) |
| **Transfer Learning** | MobileNetV2, ResNet50 | Spectrograms | O(N³) |
| **Custom** | **MaxCRNN** | Raw + Attention | O(N² log N) |

### Single-Channel Adaptations

With single-channel input (3000×1 instead of 1000×2), model architectures are adapted:

| Component | Dual-Channel | Single-Channel |
|-----------|--------------|----------------|
| Input shape | (1000, 2) | (3000, 1) |
| Inception filters | 64, 128 | 32, 64 (reduced) |
| LSTM units | 128 | 64 (reduced) |
| Total parameters | ~1.2M | ~400K |

## The MaxCRNN Architecture

### High-Level Architecture

```
Input (3000×1) → Inception Blocks → Bi-LSTM → Multi-Head Attention → Softmax
```

![Final comparison](images/final_comparison.png)
*Figure: Final strategy comparison showing all approaches tested.*

### Mathematical Foundations

#### 1. Inception Block (Multi-Scale Feature Extraction)

The Inception module (Szegedy et al., 2015) applies parallel convolutions at multiple scales:

$$
\mathbf{h}_{inc} = \text{Concat}[\mathbf{h}_{1×1}, \mathbf{h}_{3×3}, \mathbf{h}_{5×5}, \mathbf{h}_{pool}]
$$

Where each branch is:

$$
\mathbf{h}_{k×1} = \text{ReLU}(\text{Conv1D}(\mathbf{x}, \mathbf{W}_k))
$$

**Intuition:** Different kernel sizes capture temporal patterns at different scales—individual motor pulses (small kernels) vs. sustained tongue movements (large kernels).

#### 2. Bidirectional LSTM (Temporal Modeling)

The Bi-LSTM (Hochreiter & Schmidhuber, 1997) processes the sequence in both directions:

$$
\overrightarrow{\mathbf{h}_t} = \text{LSTM}(\mathbf{x}_t, \overrightarrow{\mathbf{h}_{t-1}})
$$
$$
\overleftarrow{\mathbf{h}_t} = \text{LSTM}(\mathbf{x}_t, \overleftarrow{\mathbf{h}_{t+1}})
$$
$$
\mathbf{h}_t = [\overrightarrow{\mathbf{h}_t}; \overleftarrow{\mathbf{h}_t}]
$$

**LSTM Cell Equations:**

$$
\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(Forget Gate)}
$$
$$
\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(Input Gate)}
$$
$$
\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C) \quad \text{(Candidate)}
$$
$$
\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t \quad \text{(Cell State)}
$$
$$
\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(Output Gate)}
$$
$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)
$$

**Intuition:** LSTM captures long-range temporal dependencies in the muscle activation sequence—critical for distinguishing words with similar onsets but different endings.

#### 3. Multi-Head Attention (Selective Focus)

Scaled Dot-Product Attention (Vaswani et al., 2017):

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

Multi-Head extension:

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)\mathbf{W}^O
$$

Where each head is:

$$
\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)
$$

**Intuition:** Attention allows the model to focus on the most discriminative time points (e.g., the onset of tongue movement) rather than treating all timesteps equally.

### Complete MaxCRNN Pseudocode

```
Algorithm: MaxCRNN Forward Pass (Single-Channel)
─────────────────────────────────────────────────
Input: x ∈ ℝ^(3000×1)  // Single-channel window
Output: ŷ ∈ ℝ^K        // Class probabilities

1. h₁ ← InceptionBlock(x, filters=32)   // Multi-scale features
2. h₂ ← InceptionBlock(h₁, filters=64)  // Stack 2 blocks
3. h₃ ← BiLSTM(h₂, units=64)            // Temporal modeling
4. h_attn ← MultiHeadAttention(h₃)      // Selective focus
5. h_pool ← GlobalAveragePool(h_attn)
6. ŷ ← Softmax(Dense(h_pool))
─────────────────────────────────────────────────
```

## Baseline Comparison: Random Forest

For ESP32 deployment, Random Forest remains the Pareto-optimal choice from Phase 3:

$$
G = 1 - \sum_{k=1}^{K} p_k^2 \quad \text{(Gini Impurity)}
$$

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_features='sqrt',
    bootstrap=True,
    random_state=1738
)
```

### Phase 4 Actual Results

> **Critical Finding:** All models failed. Multi-class at or below chance level (25%), binary classification collapsed to majority-class prediction.

| Model | Val Acc (L3) | Test Acc (L4) | Transfer Gap | Deployable? |
|-------|--------------|---------------|--------------|-------------|
| Random Forest (aug) | 46.67% | 22.39% | 24.28% | Useless |
| MaxCRNN | 26.67% | 23.88% | 2.79% | Useless |
| Spectrogram CNN | 30.00% | 24.38% | 5.62% | Useless |
| **Binary RF** | - | 72.64% (mode collapse) | - | **No** |

### Why Models Failed: Mode Collapse

| Model | Failure Mode | Explanation |
|-------|--------------|-------------|
| MaxCRNN | Predicted GHOST 92-94% | Collapsed to majority class |
| Spectrogram CNN | Predicted STOP 78-84% | Collapsed to single class |
| Binary RF | Predicted WORD 100% for REST | Collapsed to majority class |

### The Smoking Gun: Same-Domain Sanity Check

Even when trained AND tested on mouthing data (L3→L3), accuracy was only **27.50%**—barely above chance. This proves the signal itself lacks discriminative features.

![model_comparison.png](../working_process/colab/phase4_all_results/model_comparison.png)

## Model Selection Summary

| Use Case | Recommended Model | Rationale |
|----------|-------------------|-----------|
| **Binary Detection** | None | 72.64% is mode collapse (100% of REST → WORD) |
| **Word Classification** | None | Signal lacks discriminative info |

> **Conclusion:** For single-channel submental EMG, **no classification is viable**. All models collapsed to predicting a single class.

## References

1. Szegedy, C., et al. (2015). Going Deeper with Convolutions. *CVPR*. https://arxiv.org/abs/1409.4842

2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

3. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*. https://arxiv.org/abs/1706.03762

4. Kho, C. V. (2025). Phase 3: EMG-Based Gesture Classification with AD8232. *Minerva University CS156 Project Archive*.


---



<div style='page-break-before: always;'></div>

# Section 6: Model Training

## Overview

This section covers training procedures, cross-validation, and hyperparameter tuning for both the MaxCRNN (novel technique) and Random Forest (deployment baseline) using single-channel sEMG data.

## Training Configuration
### Hyperparameter Configuration

| Model | Parameter | Value | Rationale |
|-------|-----------|-------|-----------|
| **MaxCRNN** | Learning Rate | 0.0005 | Lower for stability with attention layers |
| | Batch Size | 64 | Memory efficient on A100 |
| | Patience | 50 | Allow convergence on small dataset |
| | Dropout | 0.3-0.5 | Prevent overfitting |
| **Random Forest** | N Estimators | 100 | Balanced accuracy/speed |
| | Max Features | √N | Standard heuristic |
| | Bootstrap | True | Variance reduction |

Phase 3 showed data augmentation boosted 1D CNN accuracy from 49.63% to **78.36%**. I apply similar techniques adapted for single-channel:




## Actual Training Results (Colab)

### MaxCRNN Training Curves

![MaxCRNN training curves](images/maxcrnn_training_curves.png)

**Observations:**
- **Loss:** Training loss flat (~1.4); validation loss increases from epoch 10 to >1.7
- **Accuracy:** Training accuracy fluctuates 20-35%; validation accuracy flat at ~23%
- **Diagnosis:** Model memorizing training noise, not learning features

### Spectrogram CNN Training (Last 5 Epochs)

```
Epoch 46/50: accuracy: 0.2737, val_accuracy: 0.2333
Epoch 47/50: accuracy: 0.2370, val_accuracy: 0.2333
Epoch 48/50: accuracy: 0.2568, val_accuracy: 0.2667
Epoch 49/50: accuracy: 0.2606, val_accuracy: 0.3000
Epoch 50/50: accuracy: 0.2814, val_accuracy: 0.3000
```

**Final Evaluation:**
- Val Accuracy (L3): 30.00%
- Test Accuracy (L4): 24.38%
- Transfer Gap: 5.62%

### Augmentation Ablation

| Condition | Test Accuracy | Change |
|-----------|---------------|--------|
| Without augmentation | 23.38% | baseline |
| With augmentation (3×) | 22.39% | **-1.00%** |

> **Finding:** Data augmentation provided no improvement and slightly hurt performance. This confirms the signal lacks features to augment—noise is noise regardless of jitter/scale.

## Resource Considerations

| Model | Training Time | GPU | Memory | Dataset Size |
|-------|---------------|-----|--------|--------------|
| **MaxCRNN** | ~30 min | A100 (recommended) | 8GB | ~200 windows |
| **MaxCRNN** | ~2 hrs | T4 | 16GB | ~200 windows |
| **Random Forest** | ~5 sec | CPU only | <1GB | ~200 windows |

---



<div style='page-break-before: always;'></div>

# Section 7: Predictions & Performance Metrics

## Overview

This section presents the out-of-sample prediction results for all evaluated models on single-channel subvocalization data. **All classifiers failed** — multi-class performed at or below chance level (25%), and binary classification collapsed to majority-class prediction.

---

## Actual Results (Transfer Learning: L3→L4)

### Multi-Class Classification (4 Classes)

| Model | Val Acc (L3) | Test Acc (L4) | Transfer Gap | vs. Chance |
|-------|--------------|---------------|--------------|------------|
| Random Forest (augmented) | 46.67% | **22.39%** | 24.28% | Worse |
| MaxCRNN | 26.67% | **23.88%** | 2.79% | Worse |
| Spectrogram CNN (MobileNetV2) | 30.00% | **24.38%** | 5.62% | Equal |
| RF (no augmentation) | - | 23.38% | - | Worse |
| Same-Domain (L3→L3) | - | 27.50% | - | Barely above |

**Chance Level:** 25% (4 classes)

> **Critical Finding:** Even the same-domain sanity check (train on L3, test on L3) only achieved 27.50% accuracy—barely above chance. This confirms the signal lacks discriminative features, not a transfer learning failure.

### Binary Classification (WORD vs REST): Also Failed

The binary classifier exhibited mode collapse — it predicted WORD for 100% of all inputs. The apparent accuracy simply reflects class imbalance (~73% WORD), not detection capability.

![Binary confusion matrix](images/binary_confusion_matrix.png)
*Binary confusion matrix: The model predicts WORD for everything, including 100% of REST samples.*

---

## Confusion Matrix Analysis

### Random Forest (L3→L4)

![RF confusion matrix](images/rf_confusion_matrix.png)

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| GHOST | 0.32 | 0.24 | 0.27 | 51 |
| LEFT | 0.21 | 0.24 | 0.23 | 50 |
| REST | 0.19 | 0.22 | 0.21 | 50 |
| STOP | 0.20 | 0.20 | 0.20 | 50 |
| **Overall** | **0.23** | **0.22** | **0.23** | **201** |

**Confusion Patterns:**
- High confusion between True REST → Predicted LEFT (36%)
- High confusion between True STOP → Predicted REST (38%)
- Near-uniform distribution across all cells (mode collapse)

---

### MaxCRNN (L3→L4)

![MaxCRNN confusion matrix](images/maxcrnn_confusion_matrix.png)

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| GHOST | 0.25 | **0.88** | 0.38 | 51 |
| LEFT | 0.00 | 0.00 | 0.00 | 50 |
| REST | 0.00 | 0.00 | 0.00 | 50 |
| STOP | 0.17 | 0.06 | 0.09 | 50 |
| **Overall** | **0.10** | **0.24** | **0.12** | **201** |

**Mode Collapse:** The model predicts GHOST for 92-94% of all inputs regardless of true class. This is a classic failure mode when the model cannot find discriminative features and defaults to the majority class prior.

---

### Spectrogram CNN (MobileNetV2)

![Spectrogram CNN confusion matrix](images/spectrogram_cnn_confusion.png)

**Results:**
- Val Accuracy (L3): 30.00%
- Test Accuracy (L4): 24.38%
- Transfer Gap: 5.62%

**Mode Collapse:** Model predominantly predicts STOP (78-84% of predictions).

---

### Same-Domain Sanity Check (L3→L3)

![Sanity check mouthing](images/sanity_check_mouthing.png)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| GHOST | 0.25 | 0.20 | 0.22 |
| LEFT | 0.33 | 0.40 | 0.36 |
| REST | 0.25 | 0.30 | 0.27 |
| STOP | 0.25 | 0.20 | 0.22 |
| **Accuracy** | | | **27.50%** |

If a model can't classify words when trained AND tested on the same high-SNR mouthing data, then the signal itself contains no discriminative information. This is not a transfer learning problem—it's a signal quality problem.

---

## Training Curves (MaxCRNN)

![MaxCRNN training curves](images/maxcrnn_training_curves.png)

**Observations:**
- **Loss:** Training loss stays flat (~1.4); validation loss increases from epoch 10, reaching >1.7
- **Accuracy:** Training accuracy fluctuates 20-35%; validation accuracy flat at ~23%
- **Diagnosis:** Model is memorizing training noise, not learning generalizable features

---

## Model Comparison
| Model | Test Acc (L4) | Train Time | Inference | Deployable |
|-------|---------------|------------|-----------|------------|
| Binary RF | 72.64% (mode collapse) | <1s | <1ms | **No** |
| Random Forest | 22.39% | <1s | <1ms | No |
| MaxCRNN | 23.88% | ~10min | ~50ms | No |
| Spectrogram CNN | 24.38% | ~5min | ~100ms | No |

---

## Final Strategy Comparison

![Final comparison](images/final_comparison.png)

| Strategy | Goal | Result | Verdict |
|----------|------|--------|---------|
| Transfer Learning (L3→L4) | 4-class words | 22-24% | Failed |
| Data Augmentation (3×) | Improve RF | -1% change | No effect |
| Extended Features (14) | Richer signal | No improvement | No effect |
| Window Overlap (50%) | More samples | No improvement | No effect |
| Spectrogram + ImageNet | Visual patterns | 24.38% | Failed |
| Binary (WORD vs REST) | Detection only | 72.64% (mode collapse) | **Failed** |

---

## Conclusions

### What the Results Tell Us

1. **Multi-class word discrimination is impossible** with single-channel data from a single electrode site
2. **Mode collapse** occurred in all models — multi-class (MaxCRNN → GHOST, SpecCNN → STOP), binary (→ WORD)
3. **Binary detection also failed** — the 72.64% accuracy is an artifact of class imbalance, not real discrimination

### Comparison to Phase 3

| Metric | Phase 3 (Forearm) | Phase 4 (Subvocal) |
|--------|-------------------|-------------------|
| Task | 3-class (CLENCH, RELAX, NOISE) | 4-class (GHOST, LEFT, STOP, REST) |
| Best Model | Random Forest (74%) | None viable |
| Success | Yes | **No** |
| Target Signal | Flexor Digitorum (large muscle) | Digastric (tiny muscle) |
| SNR | High (visible bursts) | Very Low (buried in noise) |

---

## Deployment Recommendation

**No deployment is viable.** The signal lacks discriminative information for any classification task.

| Use Case | Accuracy | Viability |
|----------|----------|-----------|
| Word Classification | ~24% | Not viable |
| Binary Detection | 72.64% (mode collapse) | Not viable |

---

*"The classifier performs at chance level because there is nothing to classify."*



---



<div style='page-break-before: always;'></div>

# Section 8: Visualization & Conclusions

## Key Conclusions

### 1. Technical Findings

| Finding | Evidence |
|---------|----------|
| **Multi-class classification failed** | 24% accuracy (chance = 25%) |
| **Binary classification also failed** | 72.64% is mode collapse — predicts WORD for 100% of REST |
| **Signal lacks discriminative info** | Per-class stats identical (mean=1921.2, std=9.7) |
| **Mode collapse in all models** | MaxCRNN → GHOST, SpecCNN → STOP, Binary → WORD |
| **Augmentation had no effect** | -1% accuracy change with 3× augmentation |

### 2. Deployment Recommendation

**No deployment is viable.** The signal lacks discriminative information for any classification task:
- Multi-class (4 words): At chance level
- Binary (WORD vs REST): Mode collapse to majority class

### 3. Limitations & Root Causes

| Limitation | Root Cause | Evidence |
|------------|------------|----------|
| Single channel | Lost spatial info (jaw vs chin) | Phase 3 worked with 1 channel because muscle is larger |
| Low SNR | AD8232 not designed for microvolt signals | Subvocal 10-100× weaker than mouthing |
| Identical per-class stats | Signal = noise + baseline; no word information | GHOST/LEFT/STOP/REST all mean=1921.2 |

---

## Visualization Gallery

### Data Quality Visualizations

![Amplitude comparison](images/viz_amplitude_comparison.png)
*Signal Amplitude Across Motor Intensity Levels: OVERT shows spike artifact; all others show flat baseline.*

![ADC distribution](images/viz_adc_distribution.png)
*ADC Distribution: Mouthing (broad) vs Subvocal (narrow spike) - indicates lower variance in target domain.*

### Random Samples per Class

![Random samples mouthing](images/viz_random_samples_mouthing.png)
*Mouthing (L3): All 4 word classes show visually indistinguishable waveforms.*

![Random samples subvocal](images/viz_random_samples_subvocal.png)
*Subvocal (L4): Similar pattern - no visible differences between word classes.*

### Spectrograms

![Spectrograms](images/viz_spectrograms.png)
*Mel-Spectrograms: All 4 classes show identical frequency content.*

### Confusion Matrices

![RF confusion matrix](images/rf_confusion_matrix.png)
*Random Forest: Near-uniform confusion (22% accuracy).*

![MaxCRNN confusion matrix](images/maxcrnn_confusion_matrix.png)
*MaxCRNN: Mode collapse to GHOST (92-94% of predictions).*

![Spectrogram CNN confusion matrix](images/spectrogram_cnn_confusion.png)
*Spectrogram CNN: Mode collapse to STOP (78-84% of predictions).*

![Binary confusion matrix](images/binary_confusion_matrix.png)
*Binary Classification: Mode collapse — 100% of REST samples predicted as WORD. The 72.64% reflects class imbalance, not detection capability.*

### Model Comparison

![Final comparison](images/final_comparison.png)
*Final Strategy Comparison: All approaches failed — multi-class at chance, binary collapsed to majority class.*

---

## Conclusion: A Negative Result

> *"The problem is not your code. The problem is your signal. And the signal is not good enough for anything."*

**This is a negative result.** The $30 hardware cannot replicate AlterEgo's functionality at any level. Single-channel submental EMG lacks the spatial resolution to discriminate tongue positions.

**To make progress, future work requires:**
- Multiple electrode sites (jaw + chin differential)
- Higher-quality instrumentation amplifiers
- Spatial feature extraction


---



<div style='page-break-before: always;'></div>

# Section 9: Executive Summary

## Project Overview

**AlterEgo's Alter Ego**: Attempting to replicate MIT Media Lab's silent speech interface for **$30** instead of **$1,200+**.

This project investigated the feasibility of word-level subvocalization classification using a single AD8232 cardiac sensor adapted for sEMG. Through rigorous experimentation across 6 machine learning strategies and 5 motor intensity levels, I discovered that **classification is not achievable with single-channel hardware**—multi-class (4 words) performed at chance level, and binary classification (WORD vs REST) collapsed to majority-class prediction.

---

## The Honest Results

### Multi-Class Classification: Failed

| Strategy | Test Accuracy (L4) | vs. Chance (25%) |
|----------|-------------------|------------------|
| Random Forest (augmented) | 22.39% | Worse |
| MaxCRNN (Inception+BiLSTM+Attention) | 23.88% | Worse |
| Spectrogram CNN (MobileNetV2) | 24.38% | Equal |
| Same-Domain Sanity Check (L3→L3) | 27.50% | Barely above |

### Binary Classification: Also Failed

The binary (WORD vs REST) classifier exhibited mode collapse — it predicted WORD for **100% of all inputs**, achieving apparent 72.64% accuracy only because ~73% of the data was WORD class. This is not detection; it's a degenerate classifier.

---

## Why Everything Failed (The Smoking Gun)

### Per-Class Signal Statistics (Mouthing Data)
```
GHOST: mean=1921.2, std=9.7
LEFT:  mean=1921.1, std=9.7
STOP:  mean=1921.2, std=9.8
REST:  mean=1921.2, std=9.8
```

> **All four word classes have identical statistics.** There is no discriminative information in the single-channel signal.

### Root Cause Analysis

1. **Hardware Limitation:** AlterEgo uses 7 electrodes across 5 sites; I had 1 electrode at 1 site
2. **Spatial Resolution Lost:** Without jaw-vs-chin differential, GHOST (tongue back) ≈ LEFT (tongue tip) electrically
3. **SNR Problem:** Subvocal signals are 10-100× smaller than mouthing; buried in AD8232 noise floor
4. **Mode Collapse:** All models collapsed to predicting dominant class (MaxCRNN → GHOST, SpecCNN → STOP, Binary → WORD)

---

## Due Diligence Summary

### What I Tried

| Approach | Rationale | Result |
|----------|-----------|--------|
| Transfer Learning (L3→L4) | Train on high-SNR mouthing, test on subvocal | Failed. The signal itself lacks features |
| Data Augmentation (3×) | Increase training diversity | No improvement (-1%) |
| Extended Features (14 features) | Add spectral, RMS, onset indicators | No improvement |
| Spectrogram + ImageNet | Visual pattern recognition | Mode collapse to single class |
| Window Overlap (50%) | More training samples | No improvement |
| Binary Simplification | Reduce to WORD vs REST | **Failed** — mode collapse to WORD |

### Data Collection Rigor

- **5 Motor Intensity Levels:** Overt → Whisper → Mouthing → Subvocal → Imagined
- **1.22M Total Samples** across all levels
- **Balanced Classes:** 24.7-25.8% per word across all levels
- **Sanity Checks:** Same-domain (L3→L3) tested before cross-domain

---

## Comparison to Phase 3 (Forearm EMG)

| Metric | Phase 3 (Forearm) | Phase 4 (Subvocal) |
|--------|-------------------|-------------------|
| Target | Grip clench | Silent words |
| Classes | 3 (CLENCH, RELAX, NOISE) | 4 (GHOST, LEFT, STOP, REST) |
| Channels | 1 | 1 |
| Best Accuracy | **74.25%** | 24.38% (chance level) |
| Deployable | Yes (Random Forest) | **No** |

**Why the difference?** The forearm flexor digitorum is a large muscle with high-amplitude signals easily captured by a single electrode. The submental muscles are tiny, produce microvolt signals, and require spatial discrimination between multiple sites to distinguish tongue positions.

---

## Conclusions

### What I Proved
1. Single-channel AD8232 **can** detect presence of muscle activation in submental region
2. Rigorous experimental methodology revealed hardware limitations before more wasted effort
3. The same-domain sanity check (27.50%) confirmed the failure is in the signal, not the transfer

### What I Disproved
1. Single-channel EMG **cannot** discriminate between phonetically distinct words
2. Transfer learning L3→L4 **does not** generalize—the source domain lacks discriminative features
3. Deep learning **cannot** extract features that don't exist in the signal
4. Binary classification **cannot** be salvaged—mode collapse shows no real WORD vs REST discrimination

### The Reality
This is a **negative result**. The $30 hardware cannot replicate AlterEgo's functionality—not even as a simplified binary trigger. Genuine subvocalization detection requires:
- Multiple electrode sites (jaw + chin minimum)
- Higher-quality instrumentation amplifiers
- Spatial feature extraction

---

## Next Steps

1. **Hardware Upgrade:** Second AD8232 for jaw-vs-chin differential (spatial features)
2. **Alternative Approach:** Test different electrode placements (masseter, temporalis)
3. **Document Learnings:** This negative result is valuable—publish to prevent others from repeating

---

*"The problem is not your code. The problem is your signal. And the signal is not good enough for anything."*


---



<div style='page-break-before: always;'></div>

# Section 10: References

## Academic Papers

### Core EMG & Signal Processing

1. **Raez, M. B. I., Hussain, M. S., & Mohd-Yasin, F. (2006).** "Techniques of EMG signal analysis: detection, processing, classification and applications." *Biological Procedures Online*, 8, 11–35.

2. **Hopkins, J. (n.d.).** "Electromyography (EMG)." *Johns Hopkins Medicine*. [Link](https://www.hopkinsmedicine.org/health/treatment-tests-and-therapies/electromyography-emg)

### AlterEgo & Silent Speech

3. **Kapur, A., Kapur, S., & Maes, P. (2018).** "AlterEgo: A Personalized Wearable Silent Speech Interface." *Proceedings of the 23rd Int. Conf. on Intelligent User Interfaces (IUI)*, 43–53.

4. **Nieto, N., et al. (2022).** "Inner speech recognition through EEG." *arXiv preprint*.

### Machine Learning Architectures

5. **Breiman, L. (2001).** "Random Forests." *Machine Learning*, 45(1), 5–32.

6. **Chen, T., & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *KDD 2016*.

7. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** "Deep Residual Learning for Image Recognition." *CVPR*, 770–778.

8. **Sandler, M., et al. (2018).** "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR*.

9. **Hochreiter, S., & Schmidhuber, J. (1997).** "Long Short-Term Memory." *Neural Computation*, 9(8), 1735–1780.

10. **Vaswani, A., et al. (2017).** "Attention Is All You Need." *NeurIPS*.

11. **Szegedy, C., et al. (2015).** "Going Deeper with Convolutions." *CVPR*.

### Hardware Documentation

12. **Analog Devices. (2013).** "AD8232: Single-Lead Heart Rate Monitor Front End." [Datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/ad8232.pdf)

13. **Espressif Systems. (2020).** "ESP32 Technical Reference Manual." [Link](https://www.espressif.com/sites/default/files/documentation/esp32_technical_reference_manual_en.pdf)

### Phase 3 Prior Work

14. **Kho, C. V. L. (2025).** "Pareto-Optimal Model Selection for Low-Cost, Single-Lead EMG Control in Embedded Systems." [GitHub](https://github.com/CarlKho-Minerva/v2-emg-muscle)

## Code Repositories

- **Phase 3 Repository:** https://github.com/CarlKho-Minerva/v2-emg-muscle
- **scikit-learn:** https://scikit-learn.org/
- **TensorFlow/Keras:** https://www.tensorflow.org/
- **librosa (Audio Processing):** https://librosa.org/

## Datasets

- **Phase 4 Dataset:** Single-channel Silent Articulation sEMG
  - **Total samples:** 1,221,527
  - **Training (L3 Mouthing):** 515,547 samples → 200 windows
  - **Testing (L4 Subvocal):** 537,901 samples → 201 windows
  - **Colab Notebook:** [Google Colab](https://colab.research.google.com/drive/1gs-ES2spTU45gKnunJ4CoxcrCUpRzlxG)

## Key Findings Summary

| Metric | Phase 3 (Forearm) | Phase 4 (Subvocal) |
|--------|-------------------|-------------------|
| Best Multi-class | 74% (RF) | 24% (failed) |
| Binary Attempt | - | 72.64% (mode collapse) |
| Conclusion | Viable | **Not viable** |

**AI Statement**: I have used Gemini 3.0 Pro and Claude Opus 4.5 (Thinking) via the Antigravity IDE to speed up the process of data collection and analysis. All interpretations and high-level decisions/analysis are done by me.

---

