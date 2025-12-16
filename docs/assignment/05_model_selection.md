# Section 5: Model Selection & Mathematical Foundations

## Overview

This section discusses model selection for **Silent Articulation classification**, including a **novel technique not covered in class**: the **MaxCRNN** (Inception + Bi-LSTM + Attention) architecture.

## Model Selection Rationale

### The "Ladder of Abstraction"

Following Phase 3's methodology, we evaluate models across increasing complexity:

| Tier | Models | Feature Set | Compute |
|------|--------|-------------|---------|
| **Heuristics** | Threshold, Variance | Raw amplitude | O(N) |
| **Classical ML** | Random Forest, SVM | Statistical (Set A) | O(N log N) |
| **Deep Learning** | 1D CNN, CRNN | Raw sequence (Set B) | O(N²) |
| **Transfer Learning** | MobileNetV2, ResNet50 | Spectrograms (Set C) | O(N³) |
| **Custom** | **MaxCRNN** | Raw + Attention | O(N² log N) |

## Novel Technique: MaxCRNN Architecture

### High-Level Architecture

```
Input (1000×2) → Inception Blocks → Bi-LSTM → Multi-Head Attention → Softmax
```

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

**Intuition:** Different kernel sizes capture temporal patterns at different scales (individual motor pulses vs. sustained contractions).

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

**Intuition:** LSTM captures long-range temporal dependencies in the muscle activation sequence.

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

**Intuition:** Attention allows the model to focus on the most discriminative time points (e.g., the onset of a tongue movement).

### Complete MaxCRNN Pseudocode

```
Algorithm: MaxCRNN Forward Pass
─────────────────────────────────────
Input: x ∈ ℝ^(1000×2)  // Dual-channel window
Output: ŷ ∈ ℝ^K        // Class probabilities

1. h₁ ← InceptionBlock(x)           // Multi-scale features
2. h₂ ← InceptionBlock(h₁)          // Stack 2 blocks
3. h₃ ← BiLSTM(h₂, units=128)       // Temporal modeling
4. h_attn ← MultiHeadAttention(h₃)  // Selective focus
5. h_pool ← GlobalAveragePool(h_attn)
6. ŷ ← Softmax(Dense(h_pool))
─────────────────────────────────────
```

## Model Initialization Code

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_maxcrnn(input_shape: tuple = (1000, 2),
                  n_classes: int = 4) -> Model:
    """
    Build the MaxCRNN architecture.

    Architecture: Inception → Bi-LSTM → Multi-Head Attention
    """
    inputs = layers.Input(shape=input_shape)

    # Inception Block 1
    x = inception_block(inputs, filters=64)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Inception Block 2
    x = inception_block(x, filters=128)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Bi-LSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Multi-Head Attention
    x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)

    # Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='MaxCRNN')


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

For ESP32 deployment, Random Forest remains the Pareto-optimal choice:

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
