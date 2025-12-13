# PHASE 4: AlterEgo's Alter Ego 🧠💬

## Subvocalization Detection with Low-Cost Hardware

> **Transfer Learning from Overt to Covert Speech using Dual-Channel Consumer ECG Hardware**

---

## 🎯 Core Objective

Replicate MIT Media Lab's **AlterEgo** system for **$30** instead of **$1,200+**.

Detect **subvocalization** (silent speech) using:
- **2x AD8232** ECG sensors (repurposed for sEMG)
- **ESP32** microcontroller @ 1000Hz sampling
- **Transfer Learning** from mouthing → subvocalization

---

## 💡 The "Hardware Hack" Insight

| System | Frequency Range | Cost |
|--------|----------------|------|
| AlterEgo (MIT) | 1.3Hz - 50Hz | $1,200+ |
| AD8232 (Consumer ECG) | 0.5Hz - 40Hz | **$12** |

**The AD8232's bandpass filter is accidentally perfect for speech EMG!**

No software filtering needed - hardware does it mechanically.

---

## 🔬 The Science: Subvocalization

**Myth:** It's purely in the brain.
**Fact:** Your brain sends signals to vocal muscles even when you don't speak.

We intercept the **firing order** before sound is produced.

### 5-Level "Descending Motor Intensity" Spectrum

| Level | Type | Signal Strength |
|-------|------|-----------------|
| 1 | **Overt Speech** | 🔊🔊🔊🔊🔊 |
| 2 | **Whisper** | 🔊🔊🔊🔊 |
| 3 | **Mouthing** (Train Here) | 🔊🔊🔊 |
| 4 | **Subvocalization** (Test Here) | 🔊🔊 |
| 5 | **Motor Imagery** | 🔊 |

**Strategy:** Train on Level 3 (huge signals), apply to Level 4 (tiny signals).

---

## 🛠 Hardware Setup

### Components (~$30 Total)

| Component | Purpose | Notes |
|-----------|---------|-------|
| **AD8232 x2** | sEMG capture | One for chin, one for jaw |
| **ESP32** | MCU @ 1000Hz | Wired serial for reliability |
| **Ag/AgCl Electrodes** | Signal pickup | Sticky foam pads only |
| **USB Power Bank** | Isolation | **NEVER use wall power!** |
| **Shielded Cable** | Noise reduction | Cut stock cable to <20cm |

### Electrode Placement

```
Channel 1 (Tongue/Articulation):
┌─────────────────────────────────────┐
│  Under-chin: Digastric/Mylohyoid    │
│  Red + Yellow: 2-3cm apart          │
│  Green: Mastoid (behind ear)        │
└─────────────────────────────────────┘

Channel 2 (Jaw/Intensity):
┌─────────────────────────────────────┐
│  Masseter muscle (cheek "socket")   │
│  Fires when you "bite down"         │
└─────────────────────────────────────┘
```

### ⚠️ Critical Hardware Notes

1. **SDN Pin Fix:** Wire SDN to 3.3V (don't leave floating)
2. **Cable Shielding:** Twist wires or wrap in foil
3. **Battery Power:** Never use wall outlet with electrodes on face
4. **Probe Distance:** Keep cables <20cm to minimize antenna effect

---

## 🗣️ Vocabulary Selection

Choose words based on **tongue gymnastics**, not meaning. Say "To do" instead of "task"

### Tier 1: High Success Rate

| Word | Muscle Activation | Signal Quality |
|------|-------------------|----------------|
| **GHOST** | Back of tongue → soft palate | ⭐⭐⭐⭐⭐ |
| **LEFT** | Tongue tip → alveolar ridge | ⭐⭐⭐⭐ |
| **STOP** | Plosive + jaw engagement | ⭐⭐⭐⭐ |
| **REST** | Baseline (silence) | Control |

### Tier 2: Direction Mapping

| Word | Phonetic Advantage |
|------|-------------------|
| LEFT | Strong "L" - tongue tip |
| RIGHT | Strong "R" - tongue curl |
| HIGH/TOP | For "Up" (harder consonants) |
| DROP | For "Down" (D + P = jaw) |

### Tier 3: Control Word

**"MAMA"** - Lips only, tongue stays flat. Use to detect noise vs. signal.

---

## 📊 Signal Processing Pipeline

```
Raw ADC → Bandpass 1-45Hz → Notch 60Hz → Epoch → Features → Classify
```

### Feature Extraction

| Type | Features |
|------|----------|
| **Time Domain** | MAV, ZCR, RMS |
| **Frequency Domain** | MFCCs → Spectrograms |

### Classification

- **Random Forest** (ESP32-deployable, 0.01ms inference)
- **CNN on Spectrograms** (mobile deployment)

---

## 🧪 Validation Protocol

### Step 1: Heartbeat Test (Sanity Check)
1. Connect AD8232 to ESP32
2. Electrodes on chest
3. Should see clean heartbeat spikes

### Step 2: Jaw Clench Test
1. Move electrodes to jaw/cheek
2. Bite down hard
3. Signal should explode

### Step 3: Subvocalization Test
1. Electrodes under chin
2. Sit still, relax face
3. Subvocalize "GHOST" forcefully
4. Look for distinct "wiggle"

---

## 📚 Key References

- **Kapur et al. (2018)** - AlterEgo: A Personalized Wearable Silent Speech Interface
- **Nieto et al. (2022)** - Inner Speech EEG protocols
- **Kho (2025)** - Phase 3 sEMG study (validating AD8232 + ESP32 + Random Forest)

---

## 🔗 Links

- [Phase 3 Paper (arXiv)](../CP-PHASE3_sEMGMuscle-arXiv_25TPE/)
- [AlterEgo Paper](./references/p43-kapur_BRjFwE6.pdf)

---

## 📝 Project Status

- [ ] Hardware acquisition (Guang Hua)
- [ ] Dual-channel wiring
- [ ] Data collection app
- [ ] Level 3 (mouthing) data collection
- [ ] Level 4 (subvocal) data collection
- [ ] Transfer learning experiment
- [ ] Real-time demo

---

*"You are building a Biological Keyboard, not a Telepathy Helmet."*
