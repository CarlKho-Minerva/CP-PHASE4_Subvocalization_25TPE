# Section 1: Data Explanation

## Dataset Overview

This project uses **single-channel surface EMG (sEMG) signals** captured from submental muscles during silent speech tasks. The data represents an extension of the Phase 3 single-lead forearm EMG dataset to silent speech recognition—an attempt to replicate MIT Media Lab's **AlterEgo** system (Kapur et al., 2018) for **$30** instead of **$1,200+**.

> **Note on Hardware Adaptation:** This project was originally designed as a dual-channel system (chin + jaw). Due to hardware limitations discovered during testing—one AD8232 exhibited ADC saturation near the 12-bit ceiling—the system was adapted to single-channel operation. Full troubleshooting documentation is available in the [working_process/](../working_process/) directory.

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
| **AD8232 x2** | Jin Hua Electronics | ~$300 each | Originally dual-channel; one unit exhibited saturation |
| **ESP32 (NodeMCU-32S)** | Jin Hua | ~$180 | MCU @ 1000Hz sampling, 3.3V logic |
| **Ag/AgCl Electrodes (5-pack)** | Medical supply | $40 | Conductive gel pads with metal snap |
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

## Hardware Adaptation: Dual-Channel to Single-Channel

### Original Design Intent

The initial design followed AlterEgo's multi-site electrode approach:
- **Channel 1 (Digastric/Mylohyoid):** Under-chin placement for tongue position tracking
- **Channel 2 (Masseter):** Jaw/cheek placement for bite intensity measurement

### Hardware Limitation Discovered

During validation testing (December 19, 2025), the two AD8232 units exhibited significantly different baseline ADC characteristics:

| Sensor | Baseline ADC | Heart LED | Operational Status |
|--------|--------------|-----------|-------------------|
| AD8232 #1 (Red PCB) | ~1,800 | Flickering ✓ | Functional |
| AD8232 #2 (Purple PCB) | ~3,800 | Not flickering | Saturation risk |

The second sensor's baseline near the 12-bit ADC ceiling (4095) meant that any muscle activation would saturate the signal, resulting in clipped waveforms and loss of amplitude information. Serial monitor output during testing:

```
ADC: 3796 | LO+: 1 | LO-: 1 | Status: ✓ Board responding
ADC: 3823 | LO+: 1 | LO-: 1 | Status: ✓ Board responding
ADC: 3921 | LO+: 1 | LO-: 1 | Status: ⚠️ ADC RAILING HIGH
```

> **[INSERT VIDEO]** [Loom Recording: Dual AD8232 Troubleshooting](https://www.loom.com/share/a893fc0e55334356979a57ffecdbcfa3)
> *Caption: Video documentation of the troubleshooting session identifying the saturation issue.*

### Design Decision: Single-Channel Focus

Given the hardware constraint, a pragmatic decision was made to proceed with single-channel data collection using the functional AD8232 unit. This decision was informed by analysis of feature discrimination capabilities:

| Feature Type | Dual-Channel | Single-Channel | Notes |
|--------------|--------------|----------------|-------|
| Spatial (chin vs jaw ratio) | ✓ Available | ✗ Lost | Cannot compare channel ratios |
| Temporal (firing sequence) | ✓ Available | ✓ Preserved | Primary discriminator |
| Frequency (ZCR, spectral) | ✓ Available | ✓ Preserved | Secondary discriminator |
| Amplitude (signal strength) | ✓ Available | ⚠ Reduced | Lower confidence without reference |

**Mitigation Strategy:** With single-channel operation, the classification model must rely primarily on **temporal features** (onset timing, duration, activation sequence) and **frequency features** (zero-crossing rate, spectral characteristics) rather than spatial discrimination between electrode sites.

Full analysis documented in: [2025-12-19_single_channel_discrimination.md](../working_process/2025-12-19_single_channel_discrimination.md)

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

### Anatomical Targets

The electrode placement targets the **Digastric/Mylohyoid muscle group**, which controls tongue elevation and depression during speech articulation.

![Digastric muscle anatomy](images/anatomy_digastric_muscle.png)
*Figure 1: Lateral view of neck musculature with digastric muscle highlighted (red). This muscle group contracts during tongue movement, producing the sEMG signal detected by under-chin electrodes. (Source: Gray's Anatomy, Public Domain, via Wikimedia Commons)*

The ground electrode is placed on the **Mastoid Process**—the bony protrusion behind the ear—which provides an electrically neutral reference point away from active musculature.

![Mastoid process anatomy](images/anatomy_mastoid_process.png)
*Figure 2: Skull showing mastoid process (red) of the temporal bone. This location is ideal for ground electrode placement due to minimal muscle activity and proximity to the signal electrodes. (Source: Wikimedia Commons, CC BY-SA 4.0)*

### Single-Channel Configuration

![Electrode placement schematic showing chin positions](images/viz_electrode_schematic.png)

**Digastric/Mylohyoid (Under-Chin) — Tongue Tracker**

| Electrode | Placement | Purpose |
|-----------|-----------|---------|
| **Signal+ (Red)** | Under-chin, left of centerline | Tongue movement detection |
| **Signal- (Yellow)** | Under-chin, right of centerline, 2-3cm apart | Differential signal |
| **Reference (Green)** | Mastoid process (behind ear) | Electrically neutral ground |

> **[INSERT IMAGE]** `images/img_electrode_placement_chin.jpg`
> *Caption: Electrode placement under the chin targeting the Digastric muscle.*

### 3.5mm Jack Wiring Mapping

Verified experimentally (see [2025-12-18_wiring_mapping_session.md](../working_process/2025-12-18_wiring_mapping_session.md)):

| 3.5mm Plug Segment | Wire Color | Body Placement |
|--------------------|------------|----------------|
| **Tip** | Yellow | Signal- (right of centerline) |
| **Ring (Middle)** | Green | Reference (mastoid) |
| **Sleeve (Back)** | Red | Signal+ (left of centerline) |

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
| 1 | **Overt Speech** | Natural speaking voice | 🔊🔊🔊🔊🔊 | Calibration |
| 2 | **Whisper** | Low-volume vocalization | 🔊🔊🔊🔊 | Calibration |
| 3 | **Mouthing** | **Open-Mouth** silent speech with maximal jaw excursion | 🔊🔊🔊 | **Training Data** (Source) |
| 4 | **Silent Articulation** | **Closed-Mouth** speech with exaggerated internal tongue movement | 🔊🔊 | **Testing Data** (Target) |
| 5 | **Imagined Speech** | Minimal/Micro-movements (Reading to self) | 🔊 | Exploratory |

### Data Collection Summary

Data was collected across all five motor intensity levels on December 19, 2025. Full session documentation: [2025-12-19_speech_spectrum_capture_session.md](../working_process/2025-12-19_speech_spectrum_capture_session.md)

| Level | Cycles | Total Samples | Output File |
|-------|--------|---------------|-------------|
| L1 Overt | 10 | 30,096 | `overt_data.csv` |
| L2 Whisper | 10 | 30,192 | `whisper_data.csv` |
| L3 Mouthing | 50 | 515,547 | `mouthing_data.csv` |
| L4 Subvocal | 51 | 537,901 | `subvocal_data.csv` |
| L5 Imagined | 10 | 107,791 | `imagined_data.csv` |
| **Total** | **131** | **1,221,527** | - |

#### Class Balance (Verified in Colab)
```
MOUTHING:  GHOST=129,354 (25.1%) | STOP=128,853 (25.0%) | LEFT=128,703 (25.0%) | REST=128,591 (24.9%)
SUBVOCAL:  GHOST=135,906 (25.3%) | REST=134,032 (24.9%) | STOP=133,993 (24.9%) | LEFT=133,970 (24.9%)
```

#### Signal Statistics (Raw ADC Values)

| Level | Mean | Std | Min | Max | Range |
|-------|------|-----|-----|-----|-------|
| OVERT | 1921.37 | 12.31 | 1 | 1989 | 1988 |
| WHISPER | 1921.02 | 8.98 | 1857 | 1987 | 130 |
| MOUTHING | 1921.18 | 9.75 | 1853 | 1991 | 138 |
| SUBVOCAL | 1921.15 | 260.62 | 22 | 192921* | 192899 |
| IMAGINED | 1921.28 | 9.55 | 1858 | 1987 | 129 |

> **⚠️ Data Anomaly:** Subvocal data contained 1 outlier sample with value 192,921 (likely a sensor glitch). This was removed during preprocessing with a `RawValue < 4000` filter.

#### Critical Finding: Per-Class Statistics (Mouthing)

| Class | Mean | Std | Range |
|-------|------|-----|-------|
| GHOST | 1921.2 | 9.7 | [1855, 1987] |
| LEFT | 1921.1 | 9.7 | [1853, 1989] |
| STOP | 1921.2 | 9.8 | [1854, 1991] |
| REST | 1921.2 | 9.8 | [1856, 1989] |

> **🔴 SMOKING GUN:** All four word classes have **identical** mean (1921.2) and standard deviation (9.7-9.8). This indicates that the single-channel signal contains **no discriminative information** for word-level classification. The signal can detect *that* muscle activation occurred, but cannot distinguish *which word* was articulated.

### Transfer Learning Rationale
**Open (Level 3) → Closed (Level 4)**

We assume that the *temporal sequence* of muscle activation (e.g., G-H-O-S-T) remains consistent between open and closed mouth states, even if the *amplitude* changes.
- **Training (Level 3):** Learn the neuromuscular "signature" of the word with high Signal-to-Noise Ratio (SNR).
- **Inference (Level 4):** Detect the same signature in the constrained, closed-mouth environment.

---

## Known Confounds and Limitations

The following artifacts were observed during data collection and documented for transparency:

| Confound | Description | Potential Impact |
|----------|-------------|------------------|
| **Saliva Swallowing** | Periodic swallowing creates spurious EMG bursts unrelated to speech | May contaminate REST class labels |
| **Post-Mouthing Muscle Tension** | After extensive L3 trials, chin muscles remain partially activated | REST labels after mouthing may not represent true baseline |
| **Syllabic Beat Artifact** | Involuntary micro-movements synchronized to word rhythm, even during L5 | "Imagined speech" may contain detectable motor artifacts |
| **Recording Timing Convention** | Words vocalized at countdown "2" to center signal in analysis window | Consistent across all levels; enables time-alignment |

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
| **Window Size** | 3 seconds per word | Countdown-aligned capture |
| **Channels** | 1 (single AD8232) | Under-chin only (hardware constraint) |
| **Power Source** | USB Battery Bank | **Safety: No wall power with face electrodes** |

---

## References

1. Kapur, A., Kapur, S., & Maes, P. (2018). AlterEgo: A personalized wearable silent speech interface. *Proceedings of the 23rd International Conference on Intelligent User Interfaces*, 43-53. https://doi.org/10.1145/3172944.3172977

2. Kho, C. V. (2025). Phase 3: EMG-Based Gesture Classification with AD8232. *Minerva University CS156 Project Archive*.

3. Analog Devices. (2012). AD8232: Single-Lead, Heart Rate Monitor Front End. *Datasheet*. https://www.analog.com/media/en/technical-documentation/data-sheets/ad8232.pdf

4. Gray, H. (1918). *Anatomy of the Human Body*. Philadelphia: Lea & Febiger. (Digastric muscle illustration, Public Domain)
