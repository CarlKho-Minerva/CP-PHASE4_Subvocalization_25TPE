# Working Process Log: Speech Spectrum Data Capture

**Date:** December 19, 2024
**Session Time:** 10:00 PM – 11:30 PM (~90 minutes)
**Activity:** Full speech spectrum data collection using custom C++ capture tools

---

## Summary

Created and deployed a suite of capture tools covering the full **Motor Intensity Spectrum** for subvocalization research. Successfully captured data across 5 levels of speech intensity.

---

## Tools Created

| Tool | Level | Description | Output File |
|------|-------|-------------|-------------|
| `capture_overt.cpp` | L1 | Speaking out loud | `overt_data.csv` |
| `capture_whisper.cpp` | L2 | Low-volume whispering | `whisper_data.csv` |
| `capture_mouthing.cpp` | L3 | Open-mouth silent (Training) | `mouthing_data.csv` |
| `capture_subvocal.cpp` | L4 | Closed-mouth articulation (Testing) | `subvocal_data.csv` |
| `capture_imagined.cpp` | L5 | Pure mental imagination | `imagined_data.csv` |

**Location:** `phase4/speech-capture/`

**Core code** copied from working Phase 3 `v2-emg-muscle/capture_guided.cpp` — only labels and instructions changed.

---

## Data Captured

| Level | Cycles | Words | Samples |
|-------|--------|-------|---------|
| L1 Overt | 10 | GHOST, LEFT, STOP, REST | ~40 |
| L2 Whisper | 10 | GHOST, LEFT, STOP, REST | ~40 |
| L3 Mouthing | 50 | GHOST, LEFT, STOP, REST | ~200 |
| L4 Subvocal | 51+ | GHOST, LEFT, STOP, REST | ~200 |
| L5 Imagined | 10 | GHOST, LEFT, STOP, REST | ~40 |

---

## ⚠️ Confounds & Limitations (Researcher Honesty)

The following artifacts and confounds were observed during data collection. These should be considered when interpreting results:

### 1. Saliva Swallowing Artifact
> **Swallowing saliva introduces unwanted muscle movement.**

Periodic swallowing during long capture sessions creates spurious EMG bursts unrelated to speech. This is unavoidable during extended recording and may contaminate REST labels.

### 2. Post-Mouthing Muscle Tension (Chin Tightness)
> **During REST phases in subvocalization, the chin muscles unconsciously tense up, likely due to fatigue from prior mouthing exercises.**

After extensive L3 (mouthing) trials, the digastric muscles seem to remain partially activated. Subject reported having to intentionally "suck in" to avoid visible tension — a "pufferfishing" reflex.

**Implication:** REST labels collected after mouthing may not represent true baseline relaxation.

### 3. Syllabic Beat Pufferfishing
> **Subject subconsciously "pufferfishes" (tenses lower chin muscles) to the syllabic beat of words.**

Even during imagined speech (L5), there appears to be involuntary micro-movement synchronized to word rhythm. This suggests true "imagined speech" may still contain detectable motor artifacts — potentially good for classification, but confounds the "pure mental" assumption.

### 4. Recording Timing Strategy
> **Words were vocalized during countdown "2" (not "1" or "3").**

To center the signal in visualizations:
- **"3"** = Preparation / Get ready
- **"2"** = **VOCALIZE NOW** (actual data)
- **"1"** = Natural decay / transition

This was consistent across all spectrum levels to enable clean time-alignment during analysis.

---

## Electrode Placement

Single-channel configuration (1× AD8232):
- **Red (Signal+):** Under-chin, left of centerline
- **Yellow (Signal-):** Under-chin, right of centerline, 2-3cm apart
- **Green (Ground):** Behind ear (mastoid process)

---

## Video Documentation

Three Loom recordings captured (see `README.md`):
1. Initial breadboard setup
2. 50 mouthing/subvocalization captures
3. Overt, whisper, imagined speech captures

---

## Next Steps

- [ ] Visualize signal differences across L1→L5
- [ ] Check if ZCR (zero-crossing rate) remains stable as amplitude decreases
- [ ] Train on L3, test on L4 (transfer learning validation)
- [ ] Address swallowing artifact in preprocessing
