# Section 7: Predictions & Performance Metrics

## Overview

This section generates out-of-sample predictions and computes performance metrics for all evaluated models.

## Prediction Generation

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def generate_predictions(model, X_test, model_type='sklearn'):
    """
    Generate predictions from trained model.

    Args:
        model: Trained classifier
        X_test: Test features
        model_type: 'sklearn' or 'keras'
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

## Expected Results (Based on Phase 3)

### Model Comparison Table

| Model | Accuracy | F1 (Target) | Latency | Deployable? |
|-------|----------|-------------|---------|-------------|
| **MaxCRNN** | ~83% | ~0.99 | 0.15ms* | No (GPU) |
| Mega Ensemble | ~78% | ~0.88 | >500ms | No (Latency) |
| 1D CNN (Aug) | ~78% | ~0.87 | 0.83ms | Yes |
| ResNet50 | ~76% | ~0.87 | >100ms | No (Latency) |
| MobileNetV2 | ~75% | ~0.86 | 9.8ms | No (RAM) |
| **Random Forest** | ~74% | ~0.81 | **0.01ms** | **Yes** |
| XGBoost | ~74% | ~0.83 | 0.01ms | Yes |
| Logistic Reg | ~68% | ~0.73 | 0.01ms | Yes |

*GPU latency on NVIDIA A100

### Transfer Learning Degradation

For Phase 4 (subvocalization), we expect ~10-15% accuracy drop from Phase 3 (mouthing) due to:
- Lower signal amplitude
- More subtle inter-word differences
- Distribution shift between training (Level 3) and test (Level 4)

## Latency Measurement

```python
import time

def measure_latency(model, X_sample, n_runs=100, model_type='sklearn'):
    """
    Measure inference latency.
    """
    # Warmup
    _ = model.predict(X_sample[:1]) if model_type == 'sklearn' else model.predict(X_sample[:1])

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        if model_type == 'sklearn':
            _ = model.predict(X_sample[:1])
        else:
            _ = model.predict(X_sample[:1])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    print(f"Latency: {mean_latency:.4f} ± {std_latency:.4f} ms")
    return mean_latency, std_latency
```

## Constraint Satisfaction Check

```python
def check_deployment_constraints(model, X_sample,
                                 max_latency_ms=100,
                                 max_memory_kb=320):
    """
    Check if model satisfies ESP32 deployment constraints.
    """
    # Latency check
    latency, _ = measure_latency(model, X_sample)
    latency_ok = latency < max_latency_ms

    # Memory check (approximate for sklearn models)
    import pickle
    model_bytes = len(pickle.dumps(model))
    model_kb = model_bytes / 1024
    memory_ok = model_kb < max_memory_kb

    print(f"Latency: {latency:.4f}ms {'✓' if latency_ok else '✗'} (limit: {max_latency_ms}ms)")
    print(f"Memory: {model_kb:.2f}KB {'✓' if memory_ok else '✗'} (limit: {max_memory_kb}KB)")

    return latency_ok and memory_ok
```
