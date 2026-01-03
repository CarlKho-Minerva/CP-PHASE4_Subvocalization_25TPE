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

