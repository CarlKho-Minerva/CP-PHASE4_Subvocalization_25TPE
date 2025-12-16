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

This tests whether models trained on high-amplitude mouthing signals can generalize to low-amplitude silent speech.

## Data Split Strategy

### Primary Split: Session-Based

To ensure realistic generalization, we use **session-based splitting** rather than random sampling:

```python
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def session_based_split(X: np.ndarray,
                        y: np.ndarray,
                        session_ids: np.ndarray,
                        test_size: float = 0.2) -> tuple:
    """
    Split data ensuring entire sessions are in train OR test.

    This prevents data leakage from temporal autocorrelation.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=1738)
    train_idx, test_idx = next(gss.split(X, y, groups=session_ids))

    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx])
```

### Secondary Split: Motor Intensity Level

For **transfer learning evaluation**:

```python
def intensity_level_split(X: np.ndarray,
                          y: np.ndarray,
                          intensity_labels: np.ndarray) -> tuple:
    """
    Split by motor intensity level for transfer learning.

    Args:
        intensity_labels: Array indicating motor level (3=mouthing, 4=subvocal)

    Returns:
        X_train (Level 3), X_test (Level 4), y_train, y_test
    """
    train_mask = intensity_labels == 3  # Mouthing
    test_mask = intensity_labels == 4   # Subvocalization

    return (X[train_mask], X[test_mask],
            y[train_mask], y[test_mask])
```

## Cross-Validation Strategy

### 5-Fold Stratified CV

For model selection and hyperparameter tuning:

```python
from sklearn.model_selection import StratifiedKFold

def get_cv_splits(X: np.ndarray,
                  y: np.ndarray,
                  n_splits: int = 5) -> StratifiedKFold:
    """
    Stratified K-Fold to maintain class proportions.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1738)
```

## Expected Data Distribution

| Split | Purpose | Expected N | Classes |
|-------|---------|------------|---------|
| Train | Model fitting | ~70% | Level 3 (Mouthing) |
| Validation | Hyperparameter tuning | ~10% | Level 3 subset |
| Test | Final evaluation | ~20% | Level 4 (Subvocalization) |

## Safety Considerations

Following Phase 3's findings, we prioritize:

1. **Precision** over Recall for the active class (avoid false positives)
2. **Latency** < 100ms for real-time control
3. **Memory** < 320KB for ESP32 deployment

The test set evaluation will focus on these deployment constraints.
