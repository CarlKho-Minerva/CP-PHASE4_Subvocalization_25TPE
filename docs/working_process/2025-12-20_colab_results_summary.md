# Phase 4: Subvocalization Classification Results

**Date:** 2025-12-20
**Environment:** Google Colab Pro (A100 GPU, High RAM)
**Data:** `speech-capture.zip` containing 5 motor intensity levels

---

## 1. Data Quality Assessment

### 1.1 Class Balance

**1️⃣ CLASS BALANCE:**

| Level | GHOST | LEFT | STOP | REST |
|-------|-------|------|------|------|
| OVERT | 28,126 (25.8%) | 26,957 (24.8%) | 26,892 (24.7%) | 26,919 (24.7%) |
| WHISPER | 27,931 (25.8%) | 26,767 (24.7%) | 26,742 (24.7%) | 26,717 (24.7%) |
| MOUTHING | 129,354 (25.1%) | 128,703 (25.0%) | 128,853 (25.0%) | 128,591 (24.9%) |
| SUBVOCAL | 135,906 (25.3%) | 133,970 (24.9%) | 133,993 (24.9%) | 134,032 (24.9%) |
| IMAGINED | 27,791 (25.8%) | 26,719 (24.8%) | 26,647 (24.7%) | 26,634 (24.7%) |

---

### 1.2 Signal Statistics (Raw ADC Values)

**2️⃣ SIGNAL STATISTICS (Raw ADC values):**

| Level | Mean | Std | Min | Max | Range |
|-------|------|-----|-----|-----|-------|
| OVERT | 1921.366375 | 12.314980 | 1 | 1989 | 1988 |
| WHISPER | 1921.019065 | 8.981001 | 1857 | 1987 | 130 |
| MOUTHING | 1921.178238 | 9.752317 | 1853 | 1991 | 138 |
| SUBVOCAL | 1921.151398 | 260.618888 | 22 | 192921 | 192899 |
| IMAGINED | 1921.283883 | 9.552761 | 1858 | 1987 | 129 |

> ⚠️ **ANOMALY:** Subvocal Max=192921 is outlier. Cleaned with `RawValue < 4000` filter (removed 1 sample).

---

### 1.3 Signal Amplitude Comparison

![viz_amplitude_comparison.png](colab/phase4_all_results/viz_amplitude_comparison.png)

**Graph Description:** Five line plots side-by-side, each showing 3,000 samples:
- **OVERT:** Green trace showing a significant downward spike (approx. 0 value)
- **WHISPER:** Blue trace with steady baseline noise
- **MOUTHING:** Purple trace with steady baseline noise
- **SUBVOCAL:** Red trace with steady baseline noise
- **IMAGINED:** Grey trace with steady baseline noise

---

### 1.4 Sample Duration Per Word

**4️⃣ SAMPLE DURATION PER WORD (Block Lengths):**

| Level | Mean (samples) | Mean (seconds) | Std | Min | Max | Total Blocks |
|-------|----------------|----------------|-----|-----|-----|--------------|
| MOUTHING | 2578 | 2.58s | 94 | 2499 | 3810 | 200 |
| SUBVOCAL | 2676 | 2.68s | 167 | 935 | 3740 | 201 |

---

### 1.5 ADC Value Distribution

![viz_adc_distribution.png](colab/phase4_all_results/viz_adc_distribution.png)

**5️⃣ ADC VALUE DISTRIBUTION (Histogram):**

**Cleanup Note:** Removed 1 outlier from Subvocal data (Values > 4000).

**Graph Description:** Two histograms side-by-side:
- **Mouthing (L3):** Blue histogram, bell-shaped distribution centered around 1921, ranging ~1880-1960
- **Subvocal (L4):** Coral histogram, very narrow tall spike centered around 1921

---

### 1.6 Per-Class Statistics (Mouthing)

**6️⃣ PER-CLASS STATISTICS (Mouthing):**

| Class | Mean | Std | Range |
|-------|------|-----|-------|
| GHOST | 1921.2 | 9.7 | [1855, 1987] |
| LEFT | 1921.1 | 9.7 | [1853, 1989] |
| STOP | 1921.2 | 9.8 | [1854, 1991] |
| REST | 1921.2 | 9.8 | [1856, 1989] |

> 🔴 **CRITICAL FINDING:** All 4 word classes have **identical statistics**. No discriminative information present.

---

## 2. Signal Visualization

### 2.1 Random Samples - Mouthing (L3)

![viz_random_samples_mouthing.png](colab/phase4_all_results/viz_random_samples_mouthing.png)

**Graph Description:** Four line charts showing normalized amplitude (y-axis -3 to 3) over 1000 samples:
- **GHOST:** Oscillatory behavior with multiple peaks
- **LEFT:** High frequency oscillation with amplitude variations
- **REST:** Oscillatory noise, similar in amplitude to word classes
- **STOP:** Distinct peaks and valleys

---

### 2.2 Random Samples - Subvocal (L4)

![viz_random_samples_subvocal.png](colab/phase4_all_results/viz_random_samples_subvocal.png)

**Graph Description:** Four line charts showing normalized amplitude over 1000 samples:
- **GHOST:** Irregular wave patterns
- **LEFT:** Sharp spikes, one large downward spike near sample 700
- **REST:** High amplitude variance with sharp peaks
- **STOP:** Initial high amplitude followed by lower amplitude noise

---

### 2.3 Mel-Spectrograms per Class

![viz_spectrograms.png](colab/phase4_all_results/viz_spectrograms.png)

**Graph Description:** Four greyscale heatmaps (Time vs Mel Bins) for GHOST, LEFT, REST, STOP. All show similar gradients from dark (top) to light (bottom) with minimal distinct features visible.

---

## 3. Sanity Check: Same-Domain Classification (L3→L3)

![sanity_check_mouthing.png](colab/phase4_all_results/sanity_check_mouthing.png)

**Results:**
- **✅ Same-Domain Accuracy (L3→L3): 0.2750** (chance = 25%)

**Classification Report (L3 only):**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| GHOST | 0.25 | 0.20 | 0.22 | 10 |
| LEFT | 0.33 | 0.40 | 0.36 | 10 |
| REST | 0.25 | 0.30 | 0.27 | 10 |
| STOP | 0.25 | 0.20 | 0.22 | 10 |
| **accuracy** | | | **0.28** | **40** |
| **macro avg** | 0.27 | 0.28 | 0.27 | 40 |

**Graph Description:** 4x4 heatmap. Highest value: **0.60** for Predicted LEFT / True REST. Diagonal (true positives): GHOST=0.20, LEFT=0.40, REST=0.30, STOP=0.20.

---

## 4. Random Forest Baseline (Transfer L3→L4)

![rf_confusion_matrix.png](colab/phase4_all_results/rf_confusion_matrix.png)

**Training:** Random Forest (200 trees, max_depth=20) with AUGMENTED data.

**Results:**
- **✅ Val Accuracy (L3):** 0.4667
- **✅ Test Accuracy (L4):** 0.2239
- **📉 Transfer Gap:** 0.2428

**Ablation (Augmented vs Non-Augmented):**
| Condition | Accuracy |
|-----------|----------|
| Without augmentation | 0.2338 |
| With augmentation | 0.2239 |
| **Improvement** | **-1.00%** |

**Classification Report (Test - L4):**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| GHOST | 0.32 | 0.24 | 0.27 | 51 |
| LEFT | 0.21 | 0.24 | 0.23 | 50 |
| REST | 0.19 | 0.22 | 0.21 | 50 |
| STOP | 0.20 | 0.20 | 0.20 | 50 |
| **accuracy** | | | **0.22** | **201** |

**Graph Description:** 4x4 blue heatmap. Diagonal: GHOST=0.24, LEFT=0.24, REST=0.22, STOP=0.20. Notable confusion: True REST→Predicted LEFT (0.36), True STOP→Predicted REST (0.38).

---

## 5. MaxCRNN (Deep Learning)

### 5.1 Training Curves

![maxcrnn_training_curves.png](colab/phase4_all_results/maxcrnn_training_curves.png)

**Training:** 200 Epochs (Early Stopping and ReduceLROnPlateau active).

**Graph Description:**
- **Loss:** Train loss starts low (~1.4), flat. Val loss increases from epoch 10, reaching >1.7.
- **Accuracy:** Train accuracy fluctuates 0.20-0.35. Val accuracy flat at ~0.23 after initial drop.

---

### 5.2 Evaluation Results

![maxcrnn_confusion_matrix.png](colab/phase4_all_results/maxcrnn_confusion_matrix.png)

**Results:**
- **Val Accuracy (L3):** 0.2667
- **Test Accuracy (L4):** 0.2388
- **Transfer Gap:** 0.0279

**Classification Report (MaxCRNN - L4):**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| GHOST | 0.25 | 0.88 | 0.38 | 51 |
| LEFT | 0.00 | 0.00 | 0.00 | 50 |
| REST | 0.00 | 0.00 | 0.00 | 50 |
| STOP | 0.17 | 0.06 | 0.09 | 50 |
| **accuracy** | | | **0.24** | **201** |

**Graph Description:** 4x4 green heatmap showing **extreme class collapse**:
- Model predicted "GHOST" for almost all inputs
- GHOST recall=0.88, but also classified 92% of LEFT, 90% of REST, 94% of STOP as GHOST
- Zero predictions for LEFT and REST classes

---

## 6. Model Comparison

![model_comparison.png](colab/phase4_all_results/model_comparison.png)

| Model | Val Accuracy (L3) | Test Accuracy (L4) | Transfer Gap |
|-------|-------------------|-------------------|--------------|
| Random Forest | 0.4667 | 0.2239 | 0.2428 |
| MaxCRNN | 0.2667 | 0.2388 | 0.0279 |

**Graph Description:** Bar chart comparing Val (Blue) vs Test (Coral):
- **Random Forest:** Large drop from Val (~47%) to Test (~22%)
- **MaxCRNN:** Both Val and Test at ~24%, low but stable

---

## 7. Advanced Strategies

### 7.1 Binary Classification (WORD vs REST)

![binary_confusion_matrix.png](colab/phase4_all_results/binary_confusion_matrix.png)

**Results:**
- Binary distribution (train): REST=42, WORD=128
- **✅ Binary Accuracy (WORD vs REST): 0.7264**

**Graph Description:** 2x2 orange heatmap:
- **True REST:** 0.00 → REST, 1.00 → WORD (all REST classified as WORD)
- **True WORD:** 0.03 → REST, 0.97 → WORD

> ⚠️ **INTERPRETATION:** High accuracy is misleading - model predicts WORD for everything (class imbalance).

---

### 7.2 Spectrogram + MobileNetV2 (Transfer Learning)

![spectrogram_cnn_confusion.png](colab/phase4_all_results/spectrogram_cnn_confusion.png)

**Results:**
- **Val Accuracy (L3):** 0.3000
- **Test Accuracy (L4):** 0.2438
- **Transfer Gap:** 0.0562

**Graph Description:** 4x4 red heatmap. Model predominantly predicts **STOP**:
- True STOP: 0.78 predicted correctly
- 78% of GHOST, 84% of LEFT, 82% of REST incorrectly predicted as STOP

---

## 8. Final Strategy Comparison

![final_comparison.png](colab/phase4_all_results/final_comparison.png)

| Strategy | Test Accuracy (L4) |
|----------|-------------------|
| Binary (WORD vs REST) | 72.64%* |
| Spectrogram CNN (MobileNetV2) | 24.38% |
| MaxCRNN | 23.88% |
| Random Forest (augmented) | 22.39% |
| RF with 50% Window Overlap | ~24% |

*Binary accuracy is misleading due to class imbalance.

---

## 9. Summary

| Metric | Value |
|--------|-------|
| Windows Created (Mouthing) | 200 |
| Windows Created (Subvocal) | 201 |
| Best Same-Domain Accuracy (L3→L3) | 27.50% |
| Best Cross-Domain Accuracy (L3→L4) | 24.38% |
| Chance Level (4 classes) | 25.00% |

**Conclusion:** All models perform at or below chance level. The signal contains no discriminative information for word classification.

---

## 10. Files Generated

```
phase4_all_results/
├── binary_confusion_matrix.png
├── final_comparison.png
├── maxcrnn_confusion_matrix.png
├── maxcrnn_training_curves.png
├── model_comparison.png
├── rf_confusion_matrix.png
├── sanity_check_mouthing.png
├── spectrogram_cnn_confusion.png
├── viz_adc_distribution.png
├── viz_amplitude_comparison.png
├── viz_random_samples_mouthing.png
├── viz_random_samples_subvocal.png
└── viz_spectrograms.png
```

---

*Verbatim transcription from Colab notebook execution - 2025-12-20*
