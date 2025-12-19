# Research Note: Single-Channel Signal Discrimination
**Date:** December 19, 2025
**Context:** Phase 4 subvocalization with one working AD8232

## The Question

With only **one channel** (under-chin), what signal characteristics can we actually rely on for word discrimination?

| Feature Type | Dual-Channel | Single-Channel |
|--------------|--------------|----------------|
| Spatial (chin vs jaw ratio) | ✅ | ❌ |
| Temporal (firing sequence) | ✅ | ✅ |
| Amplitude (signal strength) | ✅ | ⚠️ Reduced |
| Frequency (ZCR, spectral) | ✅ | ✅ |

## My Analysis

### What We Lose (Single Channel)
- **Spatial discrimination** - Can't compare chin vs jaw activity
- With dual-channel: GHOST might show "high chin, low jaw" while STOP shows "low chin, high jaw"
- With single-channel: This ratio is gone

### What We Keep

**1. Temporal Patterns (Primary)**
Each word has a distinct **sequence of muscle activation**:
- GHOST: Back-of-tongue slam → sustained → release (G-H-O-S-T)
- LEFT: Tip touch → hold → release (L-E-F-T has distinct L onset)
- STOP: Rapid plosive → brief hold (ST- is very different from -OST)

**2. Frequency Content (ZCR)**
Different tongue movements create different EMG frequency signatures:
- Fast movements = higher frequency content
- Sustained holds = lower frequency
- This is why Phase 3's Random Forest relied heavily on **Zero-Crossing Rate**

**3. Amplitude (Skepticism Valid)**
Here's where your skepticism applies:

| Word | Tongue Position | Distance to Chin Electrodes | Expected Amplitude |
|------|-----------------|-----------------------------|--------------------|
| GHOST | Back (soft palate) | Far | Lower |
| LEFT | Front (alveolar ridge) | Close | Higher |
| STOP | Mixed | Medium | Medium |

The under-chin electrodes (Digastric/Mylohyoid) **will** show amplitude differences based on tongue proximity, but with cheap hardware (AD8232, 12-bit ADC), the SNR might be too low to reliably distinguish.

## The Honest Assessment

**For single-channel with low-cost hardware, the ML model will likely rely on:**

1. **Temporal features** (~60% weight) - When things happen in the 1-second window
2. **Frequency features** (~30% weight) - ZCR, variance patterns
3. **Amplitude features** (~10% weight) - MAV, but noisy

**Comparison to AlterEgo:**
- MIT used **7 electrode sites** with high-gain custom hardware
- We're using **1 electrode site** with $13 hardware
- Our signal is ~10-100x weaker

## Mitigation Strategy

Since we're limited, maximize what we have:

1. **Exaggerated Articulation** (Level 3 Mouthing)
   - Force larger tongue movements → stronger, more distinct signals
   - This is why we train on "mouth open" before testing on "closed"

2. **Choose Words with Distinct Onsets**
   - GHOST: Velar stop (back tongue)
   - LEFT: Lateral approximant (tongue tip)
   - STOP: Alveolar stop (different from LEFT)
   - These are phonetically maximally different

3. **Feature Engineering Focus**
   - Prioritize temporal features (onset detection, duration)
   - Use windowed statistics (first 250ms vs last 250ms)
   - Less reliance on raw amplitude

## Conclusion

**You're right to be skeptical.** Single-channel does reduce discrimination power. But it's not hopeless:
- We rely on **temporal + frequency** more than amplitude
- **Exaggerated articulation** compensates for weak hardware
- The vocabulary was chosen for **maximal phonetic contrast**

Worst case: 3-4 class discrimination at ~70% accuracy (same as Phase 3 forearm)
Best case: Phase 3 methods transfer and we hit ~75%+

The real test is the data. Record some samples and let's see the signals.

---

*"The signal exists. The question is whether our hardware can find it."*
