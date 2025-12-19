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
| **Multi-class classification failed** | 24% accuracy (chance = 25%) |
| **Binary detection succeeded** | **72.64%** accuracy (WORD vs REST) |
| **Signal lacks discriminative info** | Per-class stats identical (mean=1921.2, std=9.7) |
| **Mode collapse in deep models** | MaxCRNN → GHOST 92%, SpecCNN → STOP 80% |
| **Augmentation had no effect** | -1% accuracy change with 3× augmentation |

### 2. Deployment Recommendation

For **ESP32 deployment**, use Binary Random Forest:
- **Use case:** Silence breaker / binary activation detection
- **Accuracy:** 72.64%
- **Latency:** <1ms
- **NOT viable:** Multi-word vocabulary (no discriminative signal)

### 3. Limitations & Root Causes

| Limitation | Root Cause | Evidence |
|------------|------------|----------|
| Single channel | Lost spatial info (jaw vs chin) | Phase 3 worked with 1 channel because muscle is larger |
| Low SNR | AD8232 not designed for microvolt signals | Subvocal 10-100× weaker than mouthing |
| Identical per-class stats | Signal = noise + baseline; no word information | GHOST/LEFT/STOP/REST all mean=1921.2 |

---

## Visualization Gallery

### Data Quality Visualizations

![viz_amplitude_comparison.png](../working_process/colab/phase4_all_results/viz_amplitude_comparison.png)
*Signal Amplitude Across Motor Intensity Levels: OVERT shows spike artifact; all others show flat baseline.*

![viz_adc_distribution.png](../working_process/colab/phase4_all_results/viz_adc_distribution.png)
*ADC Distribution: Mouthing (broad) vs Subvocal (narrow spike) - indicates lower variance in target domain.*

### Random Samples per Class

![viz_random_samples_mouthing.png](../working_process/colab/phase4_all_results/viz_random_samples_mouthing.png)
*Mouthing (L3): All 4 word classes show visually indistinguishable waveforms.*

![viz_random_samples_subvocal.png](../working_process/colab/phase4_all_results/viz_random_samples_subvocal.png)
*Subvocal (L4): Similar pattern - no visible differences between word classes.*

### Spectrograms

![viz_spectrograms.png](../working_process/colab/phase4_all_results/viz_spectrograms.png)
*Mel-Spectrograms: All 4 classes show identical frequency content.*

### Confusion Matrices

![rf_confusion_matrix.png](../working_process/colab/phase4_all_results/rf_confusion_matrix.png)
*Random Forest: Near-uniform confusion (22% accuracy).*

![maxcrnn_confusion_matrix.png](../working_process/colab/phase4_all_results/maxcrnn_confusion_matrix.png)
*MaxCRNN: Mode collapse to GHOST (92-94% of predictions).*

![spectrogram_cnn_confusion.png](../working_process/colab/phase4_all_results/spectrogram_cnn_confusion.png)
*Spectrogram CNN: Mode collapse to STOP (78-84% of predictions).*

![binary_confusion_matrix.png](../working_process/colab/phase4_all_results/binary_confusion_matrix.png)
*Binary Classification: 72.64% accuracy - the only success.*

### Model Comparison

![final_comparison.png](../working_process/colab/phase4_all_results/final_comparison.png)
*Final Strategy Comparison: All multi-class approaches at chance; binary succeeds.*

---

## The Pivot: From Telepathy to Clicker

> *"We are not building a 'Silent Speech Interface.' We are building a 'Biological Clicker'—a hands-free binary switch controlled by chin muscle activation."*

**Viable Product:**
- Input: Subvocalize any word
- Output: Binary trigger (On/Off)
- Use: Hands-free mouse click
- Hardware: $30 (AD8232 + ESP32)
- Accuracy: 72.64%

