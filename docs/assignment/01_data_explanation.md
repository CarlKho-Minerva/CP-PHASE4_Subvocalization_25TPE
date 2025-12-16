# Section 1: Data Explanation

## Dataset Overview

This project uses **dual-channel surface EMG (sEMG) signals** captured from facial/submental muscles during subvocalization tasks. The data represents an extension of the Phase 3 single-lead forearm EMG dataset to multi-channel silent speech recognition.

## Data Source

### Personal Digital Archive Origin
- **Creator:** Carl Vincent Ladres Kho (Minerva University)
- **Collection Period:** December 2025
- **Location:** Taipei, Taiwan
- **Context:** Final assignment for CS156 Machine Learning Pipeline

### Hardware Configuration (~$30 Total)
| Component | Purpose | Cost |
|-----------|---------|------|
| **2x AD8232** | Dual-channel sEMG capture | ~$24 |
| **ESP32** | MCU @ 1000Hz sampling | ~$6 |
| **Ag/AgCl Electrodes** | Signal pickup | ~$5 |

### Electrode Placement

**Channel 1 (Tongue/Articulation):**
- Under-chin: Digastric/Mylohyoid muscles
- Red + Yellow: 2-3cm apart
- Green: Mastoid (behind ear)

**Channel 2 (Jaw/Intensity):**
- Masseter muscle (cheek "socket")
- Captures jaw engagement during articulation

## Data Characteristics

### Classes (Based on Motor Intensity Spectrum)
| Level | Type | Signal Strength | Training/Testing |
|-------|------|-----------------|------------------|
| 3 | **Mouthing** | ⭐⭐⭐ | Training |
| 4 | **Subvocalization** | ⭐⭐ | Testing |

### Vocabulary Selection
Words were chosen based on **tongue gymnastics** (distinct muscle activations), not semantic meaning:

| Word | Muscle Activation | Signal Quality |
|------|-------------------|----------------|
| **GHOST** | Back of tongue → soft palate | ⭐⭐⭐⭐⭐ |
| **LEFT** | Tongue tip → alveolar ridge | ⭐⭐⭐⭐ |
| **STOP** | Plosive + jaw engagement | ⭐⭐⭐⭐ |
| **REST** | Baseline (silence) | Control |

## Prior Work Context

This dataset builds on **Phase 3** (Kho, 2025), which validated:
- AD8232 sensor efficacy for EMG capture
- 18 ML architecture benchmark
- Random Forest as Pareto-optimal for ESP32 deployment
- MaxCRNN achieving 99% precision on safety-critical class

## Sampling Methodology

- **Sampling Rate:** 1000Hz (satisfies Nyquist for EMG: fₛ > 2×450Hz)
- **Window Size:** 1-second non-overlapping segments
- **Protocol:** Transfer learning from overt (mouthing) to covert (subvocal) speech
