x`# Section 8: Visualization & Conclusions

## Overview

This section visualizes results and presents conclusions from the subvocalization classification pipeline.

## Confusion Matrix Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """
    Plot confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.tight_layout()
    plt.savefig(f'cm_{model_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()
```

## Training History Visualization

```python
def plot_training_history(history):
    """
    Plot training and validation curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()

    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
```

## Model Comparison Chart

```python
def plot_model_comparison(results_df):
    """
    Bar chart comparing model performance.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    models = results_df['Model']
    accuracy = results_df['Accuracy']
    f1 = results_df['F1_Score']

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, f1, width, label='F1 Score', color='coral')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()
```

## Pareto Frontier Visualization

```python
def plot_pareto_frontier(results_df):
    """
    Plot accuracy vs latency with Pareto frontier.
    """
    plt.figure(figsize=(10, 8))

    for _, row in results_df.iterrows():
        color = 'green' if row['Deployable'] else 'red'
        plt.scatter(row['Latency_ms'], row['Accuracy'],
                   c=color, s=100, alpha=0.7)
        plt.annotate(row['Model'],
                    (row['Latency_ms'], row['Accuracy']),
                    fontsize=9, ha='left')

    plt.axvline(x=100, color='orange', linestyle='--', label='Latency Limit (100ms)')
    plt.xlabel('Latency (ms) - Log Scale')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.title('Accuracy vs Latency: Pareto Frontier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pareto_frontier.png', dpi=150)
    plt.show()
```

## Key Conclusions

### 1. Technical Findings

| Finding | Evidence |
|---------|----------|
| **MaxCRNN achieves highest accuracy** | ~83% with 99% precision on target class |
| **Random Forest is Pareto-optimal for ESP32** | 74% accuracy, 0.01ms latency, <50KB |
| **Data augmentation critical for deep learning** | 29% accuracy boost (49% → 78%) |
| **Transfer learning partially succeeds** | ~10-15% drop from mouthing → subvocal |

### 2. Deployment Recommendation

For **ESP32 deployment**, use Random Forest with statistical features:
- Compile to static C++ if/else statements
- Real-time inference with negligible latency
- No runtime memory overhead

For **high-accuracy applications** (with GPU), use MaxCRNN:
- 99% precision eliminates false positives
- Suitable for safety-critical applications

### 3. Limitations & Future Work

| Limitation | Mitigation |
|------------|------------|
| Single-subject dataset | Collect from N≥10 subjects |
| Controlled environment | Real-world noise characterization |
| Binary class (mouthing vs subvocal) | Gradient of motor intensities |
| Limited vocabulary | Expand to phoneme-level recognition |

## Temporal Smoothing (Post-Processing)

```python
from collections import deque

def temporal_smoothing(predictions, window_size=5):
    """
    Apply majority vote smoothing to reduce transient errors.

    Args:
        predictions: Raw frame-level predictions
        window_size: Number of frames for majority vote
    """
    smoothed = []
    buffer = deque(maxlen=window_size)

    for pred in predictions:
        buffer.append(pred)
        if len(buffer) == window_size:
            majority = max(set(buffer), key=list(buffer).count)
            smoothed.append(majority)
        else:
            smoothed.append(pred)

    return np.array(smoothed)
```

This post-processing can boost practical reliability by rejecting transient "glitches," as demonstrated in Phase 3.
