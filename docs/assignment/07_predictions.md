# Section 7: Predictions & Performance Metrics

## Overview

This section presents the out-of-sample prediction results for all evaluated models on single-channel subvocalization data. **All classifiers failed** — multi-class performed at or below chance level (25%), and binary classification collapsed to majority-class prediction.

---

## Actual Results (Transfer Learning: L3→L4)

### Multi-Class Classification (4 Classes)

| Model | Val Acc (L3) | Test Acc (L4) | Transfer Gap | vs. Chance |
|-------|--------------|---------------|--------------|------------|
| Random Forest (augmented) | 46.67% | **22.39%** | 24.28% | Worse |
| MaxCRNN | 26.67% | **23.88%** | 2.79% | Worse |
| Spectrogram CNN (MobileNetV2) | 30.00% | **24.38%** | 5.62% | Equal |
| RF (no augmentation) | - | 23.38% | - | Worse |
| Same-Domain (L3→L3) | - | 27.50% | - | Barely above |

**Chance Level:** 25% (4 classes)

> **Critical Finding:** Even the same-domain sanity check (train on L3, test on L3) only achieved 27.50% accuracy—barely above chance. This confirms the signal lacks discriminative features, not a transfer learning failure.

### Binary Classification (WORD vs REST): Also Failed

The binary classifier exhibited mode collapse — it predicted WORD for 100% of all inputs. The apparent accuracy simply reflects class imbalance (~73% WORD), not detection capability.

![Binary confusion matrix](images/binary_confusion_matrix.png)
*Binary confusion matrix: The model predicts WORD for everything, including 100% of REST samples.*

---

## Confusion Matrix Analysis

### Random Forest (L3→L4)

![RF confusion matrix](images/rf_confusion_matrix.png)

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| GHOST | 0.32 | 0.24 | 0.27 | 51 |
| LEFT | 0.21 | 0.24 | 0.23 | 50 |
| REST | 0.19 | 0.22 | 0.21 | 50 |
| STOP | 0.20 | 0.20 | 0.20 | 50 |
| **Overall** | **0.23** | **0.22** | **0.23** | **201** |

**Confusion Patterns:**
- High confusion between True REST → Predicted LEFT (36%)
- High confusion between True STOP → Predicted REST (38%)
- Near-uniform distribution across all cells (mode collapse)

---

### MaxCRNN (L3→L4)

![MaxCRNN confusion matrix](images/maxcrnn_confusion_matrix.png)

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| GHOST | 0.25 | **0.88** | 0.38 | 51 |
| LEFT | 0.00 | 0.00 | 0.00 | 50 |
| REST | 0.00 | 0.00 | 0.00 | 50 |
| STOP | 0.17 | 0.06 | 0.09 | 50 |
| **Overall** | **0.10** | **0.24** | **0.12** | **201** |

**Mode Collapse:** The model predicts GHOST for 92-94% of all inputs regardless of true class. This is a classic failure mode when the model cannot find discriminative features and defaults to the majority class prior.

---

### Spectrogram CNN (MobileNetV2)

![Spectrogram CNN confusion matrix](images/spectrogram_cnn_confusion.png)

**Results:**
- Val Accuracy (L3): 30.00%
- Test Accuracy (L4): 24.38%
- Transfer Gap: 5.62%

**Mode Collapse:** Model predominantly predicts STOP (78-84% of predictions).

---

### Same-Domain Sanity Check (L3→L3)

![Sanity check mouthing](images/sanity_check_mouthing.png)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| GHOST | 0.25 | 0.20 | 0.22 |
| LEFT | 0.33 | 0.40 | 0.36 |
| REST | 0.25 | 0.30 | 0.27 |
| STOP | 0.25 | 0.20 | 0.22 |
| **Accuracy** | | | **27.50%** |

If a model can't classify words when trained AND tested on the same high-SNR mouthing data, then the signal itself contains no discriminative information. This is not a transfer learning problem—it's a signal quality problem.

---

## Training Curves (MaxCRNN)

![MaxCRNN training curves](images/maxcrnn_training_curves.png)

**Observations:**
- **Loss:** Training loss stays flat (~1.4); validation loss increases from epoch 10, reaching >1.7
- **Accuracy:** Training accuracy fluctuates 20-35%; validation accuracy flat at ~23%
- **Diagnosis:** Model is memorizing training noise, not learning generalizable features

---

## Model Comparison
| Model | Test Acc (L4) | Train Time | Inference | Deployable |
|-------|---------------|------------|-----------|------------|
| Binary RF | 72.64% (mode collapse) | <1s | <1ms | **No** |
| Random Forest | 22.39% | <1s | <1ms | No |
| MaxCRNN | 23.88% | ~10min | ~50ms | No |
| Spectrogram CNN | 24.38% | ~5min | ~100ms | No |

---

## Final Strategy Comparison

![Final comparison](images/final_comparison.png)

| Strategy | Goal | Result | Verdict |
|----------|------|--------|---------|
| Transfer Learning (L3→L4) | 4-class words | 22-24% | Failed |
| Data Augmentation (3×) | Improve RF | -1% change | No effect |
| Extended Features (14) | Richer signal | No improvement | No effect |
| Window Overlap (50%) | More samples | No improvement | No effect |
| Spectrogram + ImageNet | Visual patterns | 24.38% | Failed |
| Binary (WORD vs REST) | Detection only | 72.64% (mode collapse) | **Failed** |

---

## Conclusions

### What the Results Tell Us

1. **Multi-class word discrimination is impossible** with single-channel data from a single electrode site
2. **Mode collapse** occurred in all models — multi-class (MaxCRNN → GHOST, SpecCNN → STOP), binary (→ WORD)
3. **Binary detection also failed** — the 72.64% accuracy is an artifact of class imbalance, not real discrimination

### Comparison to Phase 3

| Metric | Phase 3 (Forearm) | Phase 4 (Subvocal) |
|--------|-------------------|-------------------|
| Task | 3-class (CLENCH, RELAX, NOISE) | 4-class (GHOST, LEFT, STOP, REST) |
| Best Model | Random Forest (74%) | None viable |
| Success | Yes | **No** |
| Target Signal | Flexor Digitorum (large muscle) | Digastric (tiny muscle) |
| SNR | High (visible bursts) | Very Low (buried in noise) |

---

## Deployment Recommendation

**No deployment is viable.** The signal lacks discriminative information for any classification task.

| Use Case | Accuracy | Viability |
|----------|----------|-----------|
| Word Classification | ~24% | Not viable |
| Binary Detection | 72.64% (mode collapse) | Not viable |

---

*"The classifier performs at chance level because there is nothing to classify."*

