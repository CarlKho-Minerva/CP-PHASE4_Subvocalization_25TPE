# Section 6: Model Training

## Overview

This section covers training procedures, cross-validation, and hyperparameter tuning for both the MaxCRNN (novel technique) and Random Forest (deployment baseline).

## Training Configuration

### MaxCRNN Training

```python
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_maxcrnn(model, X_train, y_train, X_val, y_val):
    """
    Train MaxCRNN with best practices from Phase 3.
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

### Hyperparameter Configuration

| Model | Parameter | Value | Rationale |
|-------|-----------|-------|-----------|
| **MaxCRNN** | Learning Rate | 0.0005 | Lower for stability |
| | Batch Size | 64 | Memory efficient |
| | Patience | 50 | Allow convergence |
| | Dropout | 0.3-0.5 | Prevent overfitting |
| **Random Forest** | N Estimators | 100 | Balanced accuracy/speed |
| | Max Features | √N | Standard heuristic |
| | Bootstrap | True | Variance reduction |

## Data Augmentation

The Phase 3 study showed data augmentation boosted 1D CNN accuracy from 49.63% to **78.36%**. We apply similar techniques:

```python
import numpy as np

def augment_window(window: np.ndarray,
                   jitter_std: float = 0.05,
                   scale_range: tuple = (0.9, 1.1),
                   shift_max: int = 50) -> np.ndarray:
    """
    Apply data augmentation to EMG window.

    Args:
        window: Shape (1000, 2) dual-channel window
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
    """
    X_aug = [X]
    y_aug = [y]

    for _ in range(augmentation_factor - 1):
        X_new = np.array([augment_window(w) for w in X])
        X_aug.append(X_new)
        y_aug.append(y)

    return np.vstack(X_aug), np.hstack(y_aug)
```

## Cross-Validation

### 5-Fold Stratified CV

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

def evaluate_with_cv(model, X, y, n_splits=5):
    """
    Evaluate model with stratified cross-validation.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1738)

    # For sklearn models
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

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

## Training Logs (Expected Output)

```
Epoch 1/1000
20/20 [==============================] - 2s 100ms/step - loss: 1.3862 - accuracy: 0.2521 - val_loss: 1.2415 - val_accuracy: 0.3200
Epoch 2/1000
20/20 [==============================] - 1s 50ms/step - loss: 1.1823 - accuracy: 0.3842 - val_loss: 1.0521 - val_accuracy: 0.4500
...
Epoch 150/1000
20/20 [==============================] - 1s 50ms/step - loss: 0.2145 - accuracy: 0.9123 - val_loss: 0.4521 - val_accuracy: 0.8321
Early stopping at epoch 150, restoring best weights from epoch 100
```

## Resource Considerations

| Model | Training Time | GPU Required | Memory |
|-------|---------------|--------------|--------|
| **MaxCRNN** | ~30 min | Yes (A100) | 8GB |
| **Random Forest** | ~5 sec | No | <1GB |
