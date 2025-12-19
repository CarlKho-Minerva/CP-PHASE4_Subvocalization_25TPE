# Section 7: Predictions & Performance Metrics

## Overview

This section generates out-of-sample predictions and computes performance metrics for all evaluated models on single-channel subvocalization data.

## Prediction Generation

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def generate_predictions(model, X_test, model_type='sklearn'):
    """
    Generate predictions from trained model.

    Args:
        model: Trained classifier
        X_test: Test features (shape depends on model)
        model_type: 'sklearn' or 'keras'

    Returns:
        y_pred: Predicted class labels
        y_proba: Prediction probabilities
    """
    if model_type == 'sklearn':
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
    else:  # Keras
        y_proba = model.predict(X_test)
        y_pred = np.argmax(y_proba, axis=1)

    return y_pred, y_proba
```

## Performance Metrics

### Metric Definitions

| Metric | Formula | Significance |
|--------|---------|--------------|
| **Accuracy** | $\frac{TP+TN}{TP+TN+FP+FN}$ | Overall correctness |
| **Precision** | $\frac{TP}{TP+FP}$ | Safety (avoid false positives) |
| **Recall** | $\frac{TP}{TP+FN}$ | Sensitivity (catch all positives) |
| **F1-Score** | $\frac{2 \cdot P \cdot R}{P + R}$ | Harmonic mean of P & R |

### Metrics Computation

```python
def compute_metrics(y_true, y_pred, class_names):
    """
    Compute comprehensive performance metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names ['GHOST', 'LEFT', 'REST', 'STOP']
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro')
    }

    # Per-class metrics
    for i, cls in enumerate(class_names):
        y_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        metrics[f'precision_{cls}'] = precision_score(y_binary, y_pred_binary, zero_division=0)
        metrics[f'recall_{cls}'] = recall_score(y_binary, y_pred_binary, zero_division=0)
        metrics[f'f1_{cls}'] = f1_score(y_binary, y_pred_binary, zero_division=0)

    return metrics


def print_classification_report(y_true, y_pred, class_names):
    """
    Print formatted classification report.
    """
    print(classification_report(y_true, y_pred, target_names=class_names))
```

## Expected Results

### Phase 4 Predictions (Transfer Learning: L3→L4)

| Model | Train Acc (L3) | Test Acc (L4) | Transfer Gap | Latency |
|-------|----------------|---------------|--------------|---------|
| **MaxCRNN** | TBD | TBD | TBD | ~50ms |
| **Random Forest** | TBD | TBD | TBD | <1ms |
| 1D CNN | TBD | TBD | TBD | ~5ms |
| SVM (RBF) | TBD | TBD | TBD | ~1ms |

> **[INSERT IMAGE]** `images/viz_model_comparison_bar.png`
> *Caption: Bar chart comparing test accuracy across all evaluated models.*

### Phase 3 Baseline Reference

| Model | Accuracy | F1 (Macro) | Latency | Deployable? |
|-------|----------|------------|---------|-------------|
| **MaxCRNN** | 83% | 0.99 | 0.15ms* | No (GPU) |
| 1D CNN (Aug) | 78% | 0.87 | 0.83ms | Yes |
| **Random Forest** | 74% | 0.81 | **0.01ms** | **Yes** |
| Logistic Reg | 68% | 0.73 | 0.01ms | Yes |

*GPU latency on NVIDIA A100

### Transfer Learning Degradation

For Phase 4 (**Silent Articulation**), we expect ~10-15% accuracy drop from training (Level 3) to test (Level 4) due to:

| Factor | Impact |
|--------|--------|
| **Lower amplitude** | L4 signals ~10x weaker than L3 |
| **Subtler differences** | Less jaw movement = smaller inter-class gap |
| **Distribution shift** | Model trained on "loud" signals, tested on "quiet" |

> **[INSERT IMAGE]** `images/viz_transfer_gap.png`
> *Caption: Training vs. test accuracy showing transfer learning degradation from L3 to L4.*

## Confusion Matrix Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    """
    Plot normalized confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
```

> **[INSERT IMAGE]** `images/viz_confusion_matrix_rf.png`
> *Caption: Confusion matrix for Random Forest showing per-class accuracy on L4 test set.*

> **[INSERT IMAGE]** `images/viz_confusion_matrix_maxcrnn.png`
> *Caption: Confusion matrix for MaxCRNN showing improved discrimination between similar words.*

### Expected Confusion Patterns

| True Class | Likely Confusion | Reason |
|------------|------------------|--------|
| GHOST | STOP | Both have plosive onset |
| LEFT | GHOST | L and G involve tongue movement |
| REST | Any | Lowest signal, noise-prone |
| STOP | LEFT | Similar tongue position |

## Latency Measurement

```python
import time

def measure_latency(model, X_sample, n_runs=100, model_type='sklearn'):
    """
    Measure inference latency.

    Args:
        model: Trained model
        X_sample: Sample input for inference
        n_runs: Number of measurement runs
        model_type: 'sklearn' or 'keras'
    """
    # Warmup
    _ = model.predict(X_sample[:1])

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_sample[:1])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    print(f"Latency: {mean_latency:.4f} ± {std_latency:.4f} ms")
    return mean_latency, std_latency
```

> **[INSERT IMAGE]** `images/viz_latency_comparison.png`
> *Caption: Latency comparison across models showing Random Forest's speed advantage.*

## Constraint Satisfaction Check

```python
def check_deployment_constraints(model, X_sample,
                                 max_latency_ms=100,
                                 max_memory_kb=320):
    """
    Check if model satisfies ESP32 deployment constraints.

    Args:
        model: Trained model
        X_sample: Sample input
        max_latency_ms: Maximum allowed latency (ESP32 real-time)
        max_memory_kb: Maximum model size (ESP32 SRAM limit)
    """
    # Latency check
    latency, _ = measure_latency(model, X_sample)
    latency_ok = latency < max_latency_ms

    # Memory check
    import pickle
    model_bytes = len(pickle.dumps(model))
    model_kb = model_bytes / 1024
    memory_ok = model_kb < max_memory_kb

    print(f"Latency: {latency:.4f}ms {'✓' if latency_ok else '✗'} (limit: {max_latency_ms}ms)")
    print(f"Memory: {model_kb:.2f}KB {'✓' if memory_ok else '✗'} (limit: {max_memory_kb}KB)")

    return latency_ok and memory_ok
```

## Per-Class Performance Summary

| Class | Expected Precision | Expected Recall | Notes |
|-------|-------------------|-----------------|-------|
| **GHOST** | High | High | Strong velar stop signal |
| **LEFT** | Medium | Medium | Tongue tip less pronounced |
| **STOP** | Medium | Medium | Plosive may confuse with GHOST |
| **REST** | Low | Variable | Baseline noise contamination |

> **[INSERT IMAGE]** `images/viz_per_class_metrics.png`
> *Caption: Per-class precision and recall comparison showing REST class challenges.*

## Prediction Confidence Analysis

```python
def analyze_confidence(y_proba, y_true, y_pred, class_names):
    """
    Analyze prediction confidence for correct vs. incorrect predictions.
    """
    max_confidence = np.max(y_proba, axis=1)

    correct_mask = y_pred == y_true
    conf_correct = max_confidence[correct_mask]
    conf_incorrect = max_confidence[~correct_mask]

    print(f"Mean confidence (correct): {np.mean(conf_correct):.4f}")
    print(f"Mean confidence (incorrect): {np.mean(conf_incorrect):.4f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.hist(conf_correct, bins=20, alpha=0.7, label='Correct', color='green')
    plt.hist(conf_incorrect, bins=20, alpha=0.7, label='Incorrect', color='red')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution: Correct vs. Incorrect Predictions')
    plt.legend()
    plt.savefig('confidence_analysis.png', dpi=150)
    plt.show()
```

> **[INSERT IMAGE]** `images/viz_confidence_distribution.png`
> *Caption: Confidence distribution showing separation between correct and incorrect predictions.*
