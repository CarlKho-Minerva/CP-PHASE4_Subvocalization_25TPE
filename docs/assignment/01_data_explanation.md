# Section 1: Data Explanation

## Dataset Overview

This project uses **dual-channel surface EMG (sEMG) signals** captured from facial/submental muscles during subvocalization tasks. The data represents an extension of the Phase 3 single-lead forearm EMG dataset to multi-channel silent speech recognition—an attempt to replicate MIT Media Lab's **AlterEgo** system for **$30** instead of **$1,200+**.

## Data Source

### Personal Digital Archive Origin
- **Creator:** Carl Vincent Ladres Kho (Minerva University)
- **Collection Period:** December 2025
- **Location:** Taipei, Taiwan (components from Guang Hua Digital Plaza)
- **Context:** Final assignment for CS156 Machine Learning Pipeline

---

## Hardware Acquisition: The Guang Hua Run

Components were purchased from **Jin Hua Electronics (今華電子)** in Guang Hua Digital Plaza, Taipei.

### Component List (~$30 USD Total)

| Component | Source | Cost (TWD) | Purpose |
|-----------|--------|------------|---------|
| **AD8232 x2** | Jin Hua Electronics | ~$300 each | Dual-channel sEMG capture |
| **ESP32 (NodeMCU-32S)** | Jin Hua | ~$180 | MCU @ 1000Hz sampling, 3.3V logic |
| **Ag/AgCl Electrodes (50-pack)** | Medical supply | ~$200 | Conductive gel pads with metal snap |
| **Shielded Audio Cable** | Jin Hua | ~$80 | Noise reduction (cut to <20cm) |
| **USB Power Bank** | Existing | - | **Safety: NEVER use wall power** |

> **[INSERT IMAGE]** `images/img_hardware_components.jpg`
> *Caption: Full hardware stack including AD8232 sensors, ESP32, and custom cables.*

### The "Accidental Hardware Match"

The **AD8232** is designed for ECG (heart monitoring), but its hardware bandpass filter is accidentally perfect for speech EMG:

| System | Target Frequency | AD8232 Filter |
|--------|------------------|---------------|
| MIT AlterEgo | 1.3Hz – 50Hz | ✓ |
| AD8232 Native | 0.5Hz – 40Hz | Hardware match! |

**No software filtering was needed—the hardware does it mechanically.**

![Frequency spectrum comparison showing AD8232 bandwidth alignment with AlterEgo requirements](images/viz_frequency_spectrum.png)

---

## Critical Hardware Fixes

### 1. The SDN Pin Fix (Discovered in Phase 3)

> **⚠️ CRITICAL:** The AD8232's Shutdown (SDN) pin floats on generic clones, causing massive signal instability.

**Solution:** Wire SDN to 3.3V (HIGH) directly on the ESP32.

```
ESP32    →    AD8232
3.3V     →    3.3V
GND      →    GND
3.3V     →    SDN (CRITICAL FIX!)
GPIO34   →    OUTPUT
```

> **[INSERT IMAGE]** `images/img_wiring_sdn_fix.jpg`
> *Caption: Close-up of the SDN pin jumping to 3.3V to prevent signal floating.*

### 2. Cable Shielding (Noise Reduction)

The stock 3-lead cable (~1 meter) acts as an antenna for 60Hz noise.

**The Fix:**
1. **Cut cable to <20cm** (face-to-board distance)
2. **Twisted Pair:** Twist Signal+ and Signal- wires together (maximizes CMRR)
3. Or use **shielded microphone cable**

> **[INSERT IMAGE]** `images/img_shielded_cable_cut.jpg`
> *Caption: Modified short-length shielded cable to minimize uptake of 60Hz mains hum.*

---

## Electrode Placement

### The "AlterEgo" Configuration

![Electrode placement schematic showing chin and jaw positions](images/viz_electrode_schematic.png)

**Channel 1: Digastric/Mylohyoid (Under-Chin) — Tongue Tracker**

| Electrode | Placement | Purpose |
|-----------|-----------|---------|
| **Signal+ (Red)** | Under-chin, left of centerline | Tongue movement detection |
| **Signal- (Yellow)** | Under-chin, right of centerline, 2-3cm apart | Differential signal |
| **Reference (Green)** | Mastoid process (behind ear) | Electrically neutral ground |

> **[INSERT IMAGE]** `images/img_electrode_placement_chin.jpg`
> *Caption: Electrode placement under the chin targeting the Digastric muscle.*

**Channel 2: Masseter (Jaw/Cheek) — Intensity Tracker**

| Electrode | Placement | Purpose |
|-----------|-----------|---------|
| **Signal+** | Fleshy part of jaw (cheek "socket") | Jaw clenching |
| **Signal-** | Slightly below, 2cm apart | Differential |
| **Reference** | Collarbone or shared with Ch1 | Ground |

> **[INSERT IMAGE]** `images/img_electrode_placement_jaw.jpg`
> *Caption: Electrode placement on the Masseter muscle for detecting jaw intensity.*

---

## Hardware Validation Protocol

To ensure signal integrity before data collection, a 3-step "Parking Lot Test" was performed:

**Step 1: Heartbeat Sanity Check**
- Electrodes on chest.
- **Success Criteria:** Clean, rhythmic QRS complex (heartbeat) every ~1s.
- **Purpose:** Verifies sensor and ADC functionality.

> **[INSERT IMAGE]** `images/img_serial_plotter_heartbeat.png`
> *Caption: Clean ECG signal confirming sensor health.*

**Step 2: Jaw Clench Noise Check**
- Electrodes on Jaw. Bite down hard.
- **Success Criteria:** Signal "explodes" into high-amplitude chaos (>2000 units).
- **Purpose:** Verifies electrodes are making contact and amplifier isn't saturated.

> **[INSERT IMAGE]** `images/img_serial_plotter_jaw_clench.png`
> *Caption: High-amplitude EMG burst during forceful jaw clench.*

**Step 3: Subvocalization "Wiggle"**
- Electrodes on Chin. Say "GHOST" internally.
- **Success Criteria:** Small but distinct disturbance from baseline noise.
- **Purpose:** Confirms detection of fine motor units in the tongue.

> **[INSERT IMAGE]** `images/img_serial_plotter_subvocal.png`
> *Caption: The "Wiggle" — subtle but distinct EMG signature of the subvocalized word "GHOST".*

---

## Vocabulary Selection: "Tongue Gymnastics"

Words were chosen based on **distinct neuromuscular signatures**, not semantic meaning.

### The Insight

> *"You are building a Biological Keyboard, not a Telepathy Helmet."*

Since electrodes are under the chin, we're tracking **tongue position**, not sound. Choose words that force the tongue to do radically different things.

### Tier 1: High Success Rate

| Word | Tongue Physics | Expected Signal |
|------|----------------|-----------------|
| **GHOST** | Back of tongue → soft palate ("G" slam) | ⭐⭐⭐⭐⭐ High-frequency burst |
| **LEFT** | Tongue tip → alveolar ridge ("L" touch) | ⭐⭐⭐⭐ Distinct onset |
| **STOP** | Plosive "ST" + "P" = jaw engagement | ⭐⭐⭐⭐ Combined signal |
| **REST** | Tongue flat, relaxed | Control (silence) |

### Tier 2: Control Word

**"MAMA"** — Lips only (Orbicularis Oris). Tongue stays flat.
- **Purpose:** If you subvocalize "MAMA" and see a chin signal spike, you're picking up **noise**, not muscle.

---

## The Motor Intensity Spectrum

### 5-Level "Descending Motor Intensity" Framework

To validate the low-cost hardware, we employ a **Transfer Learning** strategy across the motor intensity spectrum.

> **The Insight:** Training on "Open Mouth" movements (Mouthing) provides strong, high-amplitude signals that help the model learn the temporal dynamics of each word. We then transfer this knowledge to "Closed Mouth" (Silent Articulation) scenarios.

| Level | Terminology | Description | Signal | Role |
|-------|-------------|-------------|--------|------|
| 1 | **Overt Speech** | Natural speaking voice | 🔊🔊🔊🔊🔊 | — |
| 2 | **Whisper** | Low-volume vocalization | 🔊🔊🔊🔊 | Calibration |
| 3 | **Mouthing** | **Open-Mouth** silent speech with maximal jaw excursion | 🔊🔊🔊 | **Training Data** (Source) |
| 4 | **Silent Articulation** | **Closed-Mouth** speech with exaggerated internal tongue movement | 🔊🔊 | **Testing Data** (Target) |
| 5 | **Subvocalization** | Minimal/Micro-movements (Reading to self) | 🔊 | Future Work |

### Transfer Learning Rationale
**Open (Level 3) → Closed (Level 4)**

We assume that the *temporal sequence* of muscle activation (e.g., G-H-O-S-T) remains consistent between open and closed mouth states, even if the *amplitude* changes.
- **Training (Level 3):** Learn the neuromuscular "signature" of the word with high Signal-to-Noise Ratio (SNR).
- **Inference (Level 4):** Detect the same signature in the constrained, closed-mouth environment.


---

## Prior Work Context

This dataset builds on **Phase 3** (Kho, 2025), which validated:
- AD8232 sensor efficacy for EMG capture (SDN pin fix discovered)
- 18 ML architecture benchmark on 1.54M data points
- Random Forest as Pareto-optimal for ESP32 deployment (74% accuracy, 0.01ms)
- MaxCRNN achieving 99% precision on safety-critical class

---

## Sampling Methodology

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Sampling Rate** | 1000Hz | Nyquist: fₛ > 2×450Hz (EMG bandwidth) |
| **ADC Resolution** | 12-bit (0-4095) | ESP32 native |
| **Window Size** | 1 second (1000 samples) | Non-overlapping |
| **Channels** | 2 (dual AD8232) | Chin + Jaw |
| **Power Source** | USB Battery Bank | **Safety: No wall power with face electrodes** |

