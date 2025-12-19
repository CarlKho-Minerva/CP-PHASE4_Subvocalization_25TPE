# Section 5: Model Selection & Mathematical Foundations

## Overview

This section discusses model selection for **Silent Articulation classification** from single-channel sEMG signals, including a **novel technique not covered in class**: the **MaxCRNN** (Inception + Bi-LSTM + Attention) architecture.

## Model Selection Rationale

### The "Ladder of Abstraction"

Following Phase 3's methodology, we evaluate models across increasing complexity:

| Tier | Models | Feature Set | Compute |
|------|--------|-------------|---------|
| **Heuristics** | Threshold, Variance | Raw amplitude | O(N) |
| **Classical ML** | Random Forest, SVM | Statistical (MAV, ZCR, SD, MAX) | O(N log N) |
| **Deep Learning** | 1D CNN, CRNN | Raw sequence | O(N²) |
| **Transfer Learning** | MobileNetV2, ResNet50 | Spectrograms | O(N³) |
| **Custom** | **MaxCRNN** | Raw + Attention | O(N² log N) |

> **[INSERT IMAGE]** `images/viz_model_ladder.png`
> *Caption: Model complexity ladder from simple heuristics to custom deep learning architectures.*

### Single-Channel Adaptations

With single-channel input (3000×1 instead of 1000×2), model architectures are adapted:

| Component | Dual-Channel | Single-Channel |
|-----------|--------------|----------------|
| Input shape | (1000, 2) | (3000, 1) |
| Inception filters | 64, 128 | 32, 64 (reduced) |
| LSTM units | 128 | 64 (reduced) |
| Total parameters | ~1.2M | ~400K |

## Novel Technique: MaxCRNN Architecture

### High-Level Architecture

```
Input (3000×1) → Inception Blocks → Bi-LSTM → Multi-Head Attention → Softmax
```

> **[INSERT IMAGE]** `images/viz_maxcrnn_architecture.png`
> *Caption: MaxCRNN architecture diagram showing Inception blocks, Bi-LSTM, and attention layers.*

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

> **[INSERT IMAGE]** `images/viz_inception_block.png`
> *Caption: Inception block showing parallel 1×1, 3×3, 5×5 convolutions and max pooling branch.*

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

> **[INSERT IMAGE]** `images/viz_attention_weights.png`
> *Caption: Visualization of attention weights showing focus on word onset and offset regions.*

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

## Model Initialization Code

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_maxcrnn(input_shape: tuple = (3000, 1),
                  n_classes: int = 4) -> Model:
    """
    Build the MaxCRNN architecture for single-channel sEMG.

    Architecture: Inception → Bi-LSTM → Multi-Head Attention
    """
    inputs = layers.Input(shape=input_shape)

    # Inception Block 1
    x = inception_block(inputs, filters=32)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Inception Block 2
    x = inception_block(x, filters=64)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Bi-LSTM (reduced units for single-channel)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    # Multi-Head Attention
    x = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)

    # Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='MaxCRNN_SingleChannel')


def inception_block(x, filters: int):
    """
    1D Inception block with parallel convolutions.
    """
    conv1 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
    conv3 = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
    conv5 = layers.Conv1D(filters, 5, padding='same', activation='relu')(x)
    pool = layers.MaxPooling1D(3, strides=1, padding='same')(x)
    pool = layers.Conv1D(filters, 1, padding='same', activation='relu')(pool)

    return layers.Concatenate()([conv1, conv3, conv5, pool])
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

> ⚠️ **Critical Finding:** All multi-class models performed at or below chance level (25%). Only binary classification succeeded.

| Model | Val Acc (L3) | Test Acc (L4) | Transfer Gap | Deployable? |
|-------|--------------|---------------|--------------|-------------|
| Random Forest (aug) | 46.67% | 22.39% | 24.28% | ❌ Useless |
| MaxCRNN | 26.67% | 23.88% | 2.79% | ❌ Useless |
| Spectrogram CNN | 30.00% | 24.38% | 5.62% | ❌ Useless |
| **Binary RF** | - | **72.64%** | - | ✅ Yes |

### Why Models Failed: Mode Collapse

| Model | Failure Mode | Explanation |
|-------|--------------|-------------|
| MaxCRNN | Predicted GHOST 92-94% | Collapsed to majority class |
| Spectrogram CNN | Predicted STOP 78-84% | Collapsed to single class |
| Random Forest | Near-uniform confusion | No features to learn |

### The Smoking Gun: Same-Domain Sanity Check

Even when trained AND tested on mouthing data (L3→L3), accuracy was only **27.50%**—barely above chance. This proves the signal itself lacks discriminative features.

![model_comparison.png](../working_process/colab/phase4_all_results/model_comparison.png)

## Model Selection Summary

| Use Case | Recommended Model | Rationale |
|----------|-------------------|-----------|
| **Binary Detection** | Random Forest | 72.64% accuracy, <1ms latency |
| **Word Classification** | None | Signal lacks discriminative info |

> **Conclusion:** For single-channel submental EMG, the only viable product is a **binary "Silence Breaker" switch**, not a multi-word vocabulary interface.

## References

1. Szegedy, C., et al. (2015). Going Deeper with Convolutions. *CVPR*. https://arxiv.org/abs/1409.4842

2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

3. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*. https://arxiv.org/abs/1706.03762

4. Kho, C. V. (2025). Phase 3: EMG-Based Gesture Classification with AD8232. *Minerva University CS156 Project Archive*.
