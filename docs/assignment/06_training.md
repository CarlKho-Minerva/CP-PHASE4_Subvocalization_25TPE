# Section 6: Model Training

## Overview

This section covers training procedures, cross-validation, and hyperparameter tuning for both the MaxCRNN (novel technique) and Random Forest (deployment baseline) using single-channel sEMG data.

## Training Configuration

### MaxCRNN Training

```python
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def train_maxcrnn(model, X_train, y_train, X_val, y_val):
    """
    Train MaxCRNN with best practices from Phase 3.

    Args:
        model: Compiled MaxCRNN model
        X_train: Training windows, shape (N, 3000, 1)
        y_train: Training labels, shape (N,)
        X_val: Validation windows
        y_val: Validation labels
    """
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            'best_maxcrnn.keras',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1000,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    return history
```

> **[INSERT IMAGE]** `images/viz_training_curves.png`
> *Caption: Training and validation loss/accuracy curves showing convergence.*

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

## Data Augmentation

Phase 3 showed data augmentation boosted 1D CNN accuracy from 49.63% to **78.36%**. We apply similar techniques adapted for single-channel:

```python
import numpy as np

def augment_window(window: np.ndarray,
                   jitter_std: float = 0.05,
                   scale_range: tuple = (0.9, 1.1),
                   shift_max: int = 100) -> np.ndarray:
    """
    Apply data augmentation to single-channel EMG window.

    Args:
        window: Shape (3000, 1) single-channel window
        jitter_std: Gaussian noise standard deviation
        scale_range: Amplitude scaling range
        shift_max: Maximum time shift (samples)
    """
    augmented = window.copy()

    # 1. Jitter: Add Gaussian noise
    noise = np.random.normal(0, jitter_std, window.shape)
    augmented = augmented + noise

    # 2. Scaling: Random amplitude multiplier
    scale = np.random.uniform(*scale_range)
    augmented = augmented * scale

    # 3. Time Shift: Circular shift
    shift = np.random.randint(-shift_max, shift_max)
    augmented = np.roll(augmented, shift, axis=0)

    return augmented


def create_augmented_dataset(X: np.ndarray,
                             y: np.ndarray,
                             augmentation_factor: int = 5) -> tuple:
    """
    Create augmented training set.

    Args:
        X: Original windows, shape (N, 3000, 1)
        y: Original labels, shape (N,)
        augmentation_factor: Number of augmented copies (including original)

    Returns:
        X_aug: Augmented windows, shape (N*factor, 3000, 1)
        y_aug: Augmented labels, shape (N*factor,)
    """
    X_aug = [X]
    y_aug = [y]

    for _ in range(augmentation_factor - 1):
        X_new = np.array([augment_window(w) for w in X])
        X_aug.append(X_new)
        y_aug.append(y)

    return np.vstack(X_aug), np.hstack(y_aug)
```

> **[INSERT IMAGE]** `images/viz_augmentation_examples.png`
> *Caption: Examples of original vs. augmented EMG windows showing jitter, scaling, and time shift effects.*

## Cross-Validation

### 5-Fold Stratified CV

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

def evaluate_with_cv(model, X, y, n_splits=5):
    """
    Evaluate model with stratified cross-validation.

    Args:
        model: Sklearn-compatible classifier
        X: Feature matrix, shape (N, 4) for statistical features
        y: Labels, shape (N,)
        n_splits: Number of CV folds
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1738)

    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune_random_forest(X_train, y_train):
    """
    Grid search for Random Forest hyperparameters.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestClassifier(random_state=1738)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_
```

> **[INSERT IMAGE]** `images/viz_hyperparameter_search.png`
> *Caption: Grid search results showing accuracy across hyperparameter combinations.*

## Actual Training Results (Colab)

### MaxCRNN Training Curves

![maxcrnn_training_curves.png](../working_process/colab/phase4_all_results/maxcrnn_training_curves.png)

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

> ⚠️ **Finding:** Data augmentation provided no improvement and slightly hurt performance. This confirms the signal lacks features to augment—noise is noise regardless of jitter/scale.

## Transfer Learning Metrics

```python
class TransferMetricsCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to monitor performance on target domain during training.
    """
    def __init__(self, X_target, y_target):
        self.X_target = X_target
        self.y_target = y_target

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            y_pred = self.model.predict(self.X_target, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            target_acc = np.mean(y_pred_classes == self.y_target)
            print(f"\n  → Target Domain (L4) Accuracy: {target_acc:.4f}")
```

> **[INSERT IMAGE]** `images/viz_transfer_learning_gap.png`
> *Caption: Training curve showing source (L3) vs. target (L4) accuracy gap over epochs.*

## Resource Considerations

| Model | Training Time | GPU | Memory | Dataset Size |
|-------|---------------|-----|--------|--------------|
| **MaxCRNN** | ~30 min | A100 (recommended) | 8GB | ~200 windows |
| **MaxCRNN** | ~2 hrs | T4 | 16GB | ~200 windows |
| **Random Forest** | ~5 sec | CPU only | <1GB | ~200 windows |

### Colab Pro Configuration

```python
# Verify A100 GPU
!nvidia-smi

# Expected output:
# Tesla A100-SXM4-40GB

# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

## Checkpointing and Model Export

```python
# Save best model
model.save('maxcrnn_phase4_final.keras')

# Export for TensorFlow Lite (ESP32 deployment)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('maxcrnn_phase4.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model size: {len(tflite_model) / 1024:.1f} KB")
```

> **[INSERT IMAGE]** `images/viz_model_size_comparison.png`
> *Caption: Model size comparison showing Keras vs. TFLite optimized versions.*
