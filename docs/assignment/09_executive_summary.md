# Section 9: Executive Summary

## Project Overview

**AlterEgo's Alter Ego**: Attempting to replicate MIT Media Lab's silent speech interface for **$30** instead of **$1,200+**.

This project investigated the feasibility of word-level subvocalization classification using a single AD8232 cardiac sensor adapted for sEMG. Through rigorous experimentation across 6 machine learning strategies and 5 motor intensity levels, I discovered that **classification is not achievable with single-channel hardware**—multi-class (4 words) performed at chance level, and binary classification (WORD vs REST) collapsed to majority-class prediction.

---

## The Honest Results

### Multi-Class Classification: Failed

| Strategy | Test Accuracy (L4) | vs. Chance (25%) |
|----------|-------------------|------------------|
| Random Forest (augmented) | 22.39% | Worse |
| MaxCRNN (Inception+BiLSTM+Attention) | 23.88% | Worse |
| Spectrogram CNN (MobileNetV2) | 24.38% | Equal |
| Same-Domain Sanity Check (L3→L3) | 27.50% | Barely above |

### Binary Classification: Also Failed

The binary (WORD vs REST) classifier exhibited mode collapse — it predicted WORD for **100% of all inputs**, achieving apparent 72.64% accuracy only because ~73% of the data was WORD class. This is not detection; it's a degenerate classifier.

---

## Why Everything Failed (The Smoking Gun)

### Per-Class Signal Statistics (Mouthing Data)
```
GHOST: mean=1921.2, std=9.7
LEFT:  mean=1921.1, std=9.7
STOP:  mean=1921.2, std=9.8
REST:  mean=1921.2, std=9.8
```

> **All four word classes have identical statistics.** There is no discriminative information in the single-channel signal.

### Root Cause Analysis

1. **Hardware Limitation:** AlterEgo uses 7 electrodes across 5 sites; I had 1 electrode at 1 site
2. **Spatial Resolution Lost:** Without jaw-vs-chin differential, GHOST (tongue back) ≈ LEFT (tongue tip) electrically
3. **SNR Problem:** Subvocal signals are 10-100× smaller than mouthing; buried in AD8232 noise floor
4. **Mode Collapse:** All models collapsed to predicting dominant class (MaxCRNN → GHOST, SpecCNN → STOP, Binary → WORD)

---

## Due Diligence Summary

### What I Tried

| Approach | Rationale | Result |
|----------|-----------|--------|
| Transfer Learning (L3→L4) | Train on high-SNR mouthing, test on subvocal | Failed. The signal itself lacks features |
| Data Augmentation (3×) | Increase training diversity | No improvement (-1%) |
| Extended Features (14 features) | Add spectral, RMS, onset indicators | No improvement |
| Spectrogram + ImageNet | Visual pattern recognition | Mode collapse to single class |
| Window Overlap (50%) | More training samples | No improvement |
| Binary Simplification | Reduce to WORD vs REST | **Failed** — mode collapse to WORD |

### Data Collection Rigor

- **5 Motor Intensity Levels:** Overt → Whisper → Mouthing → Subvocal → Imagined
- **1.22M Total Samples** across all levels
- **Balanced Classes:** 24.7-25.8% per word across all levels
- **Sanity Checks:** Same-domain (L3→L3) tested before cross-domain

---

## Comparison to Phase 3 (Forearm EMG)

| Metric | Phase 3 (Forearm) | Phase 4 (Subvocal) |
|--------|-------------------|-------------------|
| Target | Grip clench | Silent words |
| Classes | 3 (CLENCH, RELAX, NOISE) | 4 (GHOST, LEFT, STOP, REST) |
| Channels | 1 | 1 |
| Best Accuracy | **74.25%** | 24.38% (chance level) |
| Deployable | Yes (Random Forest) | **No** |

**Why the difference?** The forearm flexor digitorum is a large muscle with high-amplitude signals easily captured by a single electrode. The submental muscles are tiny, produce microvolt signals, and require spatial discrimination between multiple sites to distinguish tongue positions.

---

## Conclusions

### What I Proved
1. Single-channel AD8232 **can** detect presence of muscle activation in submental region
2. Rigorous experimental methodology revealed hardware limitations before more wasted effort
3. The same-domain sanity check (27.50%) confirmed the failure is in the signal, not the transfer

### What I Disproved
1. Single-channel EMG **cannot** discriminate between phonetically distinct words
2. Transfer learning L3→L4 **does not** generalize—the source domain lacks discriminative features
3. Deep learning **cannot** extract features that don't exist in the signal
4. Binary classification **cannot** be salvaged—mode collapse shows no real WORD vs REST discrimination

### The Reality
This is a **negative result**. The $30 hardware cannot replicate AlterEgo's functionality—not even as a simplified binary trigger. Genuine subvocalization detection requires:
- Multiple electrode sites (jaw + chin minimum)
- Higher-quality instrumentation amplifiers
- Spatial feature extraction

---

## Next Steps

1. **Hardware Upgrade:** Second AD8232 for jaw-vs-chin differential (spatial features)
2. **Alternative Approach:** Test different electrode placements (masseter, temporalis)
3. **Document Learnings:** This negative result is valuable—publish to prevent others from repeating

---

*"The problem is not your code. The problem is your signal. And the signal is not good enough for anything."*
