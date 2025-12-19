# Section 4: Analysis Plan & Data Splits

## Overview

This section describes the classification task and train/test split methodology for the subvocalization pipeline.

## Classification Task

### Problem Definition

**Multi-class classification** of subvocalized words from dual-channel sEMG signals.

| Aspect | Description |
|--------|-------------|
| **Task Type** | Multi-class Classification |
| **Input** | Dual-channel sEMG window (1000×2) |
| **Output** | Word class (GHOST, LEFT, STOP, REST, etc.) |
| **Metric** | Accuracy, Precision, F1-Score |

### Transfer Learning Strategy

The key insight of Phase 4 is **transfer learning from overt to covert speech**:

```
Level 3 (Mouthing) → Train
        ↓
Level 4 (Subvocalization) → Test
```

**Multi-class classification** of silent speech words from dual-channel sEMG signals.

### Target Classes (4 classes)
- **GHOST**
- **LEFT**
- **STOP**
- **REST** (Null class)

*(Note: "MAMA" is used only for hardware validation, not classification)*

## Data Split Strategy

### Primary Split: Session-Based

To ensure realistic generalization, we use **session-based splitting** rather than random sampling:

## Train/Test Split Methodology

### Strategy: Transfer Learning across Motor Intensities

The core hypothesis is that models trained on **Mouthing (Open Articulation)** can generalize to **Silent Articulation (Closed Articulation)**.

> **Why this matters:** We assume the *temporal sequence* of muscle activation is consistent between Open and Closed states, even if the *amplitude* differs by an order of magnitude.

| Split | Percentage | Data Source | Rationale |
|-------|------------|-------------|-----------|
| **Train** | ~70% | **Level 3: Mouthing (Open Mouth)** | **Source Domain:** High-amplitude, exaggerated signals to learn temporal dynamics. |
| **Validation** | ~10% | **Level 3: Mouthing (Open Mouth)** | Hyperparameter tuning on clean source data. |
| **Test** | ~20% | **Level 4: Silent Articulation (Closed Mouth)** | **Target Domain:** Low-amplitude, constrained signals (Real-world scenario). |

### Implementation in Code

```python
# Conceptual splitting logic
def create_transfer_splits(X, y, intensity_labels):
    """
    Splits data based on motor intensity level.
    Args:
        X, y: Features and labels
        intensity_labels: Array indicating motor level (3=Mouthing, 4=Silent Articulation)
    """
    # Train heavily on Source Domain (Level 3 - Open)
    train_mask = intensity_labels == 3
    X_train = X[train_mask]
    y_train = y[train_mask]

    # Test strictly on Target Domain (Level 4 - Closed)
    test_mask = intensity_labels == 4
    X_test = X[test_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test
```

## Evaluation Metrics

1. **Accuracy:** Overall correctness across all 4 classes.
2. **F1-Score (Macro):** Balanced metric accounting for class imbalances (if any).
3. **Confusion Matrix:** To visualize specific misclassifications (e.g., confusing "GHOST" with "STOP").
4. **Inference Latency:** Must be <5ms per window for real-time viability on ESP32.

### Success Criteria
- **Baseline:** >60% accuracy on Test set (Level 4).
- **Target:** >80% accuracy on Test set (Level 4).
- **Latency:** <5ms inference time.

## Analysis Pipeline Steps

1. **Data Loading:** Load CSVs, segment into 1s windows.
2. **Preprocessing:** Bandpass (calculated in hardware), Notch (60Hz), Standardization.
3. **Feature Engineering:** Extract statistical features (RMS, MAV, ZC, WL) and raw sequences.
4. **Model Training:**
    - Baseline: Random Forest (proven Pareto-optimal in Phase 3).
    - Deep Learning: MaxCRNN (for maximum precision).
5. **Evaluation:** Compute metrics on the "Silent Articulation" test set.
6. **Visualization:** Plot confusion matrices and feature distributions.

## Safety Considerations

Following Phase 3's findings, we prioritize:

1. **Precision** over Recall for the active class (avoid false positives)
2. **Latency** < 100ms for real-time control
3. **Memory** < 320KB for ESP32 deployment

The test set evaluation will focus on these deployment constraints.
