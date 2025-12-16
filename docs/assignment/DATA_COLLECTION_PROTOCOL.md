# MIT Media Lab "AlterEgo" Replication: Data Collection Protocol

## 1. Research Rationale (The "Why")
To rigorously validate the **"Train Loud, Predict Quiet"** hypothesis, we must capture the full **Motor Intensity Spectrum**. Collecting data across all 4 levels (Overt, Whisper, Mouthing, Silent) allows us to:
1.  **Prove the Manifold:** Show that class clusters drift linearly from L1 → L4.
2.  **Validate Calibration:** Use L1/L2 as high-SNR anchors to verify signal quality before crucial L3/L4 collection.
3.  **Enable Curriculum Learning:** Ideally, we might even train on L3+L2 to help the model find the "direction" of the silence.

## 2. Directory Structure (The "Where")
Save CSV files in this exact hierarchy to make the Colab notebook's `glob` patterns work automatically.

```text
/content/drive/MyDrive/Phase4_Data/
└── Subject_01/
    ├── Session_01_Morning/
    │   ├── Level_1_Overt/
    │   │   ├── GHOST_01.csv ... GHOST_50.csv
    │   │   ├── LEFT_01.csv ...
    │   │   ├── STOP_01.csv ...
    │   │   └── REST_01.csv ...
    │   ├── Level_2_Whisper/
    │   ├── Level_3_Mouthing/      <-- PRIMARY TRAINING SET
    │   └── Level_4_Silent/        <-- PRIMARY TESTING SET
    └── Session_02_Evening/
        └── ... (Repeated Structure)
```

## 3. Sampling Targets (The "How Many")

> **Golden Rule:** 50 clean repetitions per class, per level.

| Class | L1 (Overt) | L2 (Whisper) | L3 (Mouthing) | L4 (Silent) | **Total** |
|-------|------------|--------------|---------------|-------------|-----------|
| **GHOST** | 10 | 10 | **50** | **50** | 120 |
| **LEFT** | 10 | 10 | **50** | **50** | 120 |
| **STOP** | 10 | 10 | **50** | **50** | 120 |
| **REST** | 10 | 10 | **50** | **50** | 120 |
| **Total** | 40 | 40 | **200** | **200** | **480** |

*Estimated Time:* 480 seconds = 8 minutes of raw recording + setup time ≈ **30 minutes total**.

## 4. Collection Procedure (Step-by-Step)

### A. Setup & Calibration (The "Parking Lot")
1.  **Placement:** Digastric (Chin) + Masseter (Jaw).
2.  **Sanity Check:** Do the "Heartbeat Check" (electrodes on chest) to verify serial is plotting clean QRS complexes.
3.  **Noise Check:** Subvocalize "MAMA". If you see a spike, **your gain is too high** or electrodes are loose. Recalibrate until "MAMA" is flat.

### B. The "Descension" Protocol
Do NOT jump straight to silence. Warm up the neuromuscular pathway.

1.  **Start with Level 1 (Overt):** Say "GHOST" out loud 10 times. Watch the plotter. Ensure distinct bursts.
2.  **Fade to Level 2 (Whisper):** Whisper "GHOST" 10 times. Signal should drop ~50%.
3.  **The Main Event - Level 3 (Mouthing):**
    - Mute your voice.
    - Exaggerate the jaw drop on "G-HOST".
    - Exaggerate the tongue tap on "L-EFT".
    - **Capture 50 reps.** This is your **Training Data**.
4.  **The Challenge - Level 4 (Silent):**
    - Close your lips (lightly).
    - Perform the *exact same* tongue gymnastics as Level 3, but constrained inside the mouth.
    - **Capture 50 reps.** This is your **Testing Data**.

## 5. File Naming Convention
The Python script will parse filenames for labels. Use this format:
`LABEL_RepetitionID.csv`

Examples:
- `GHOST_01.csv`
- `LEFT_42.csv`
- `REST_05.csv` (Record empty silence for 1s)

## 6. Colab Preparation
1.  Zip your entire `Phase4_Data` folder.
2.  Upload `Phase4_Data.zip` to your Google Drive root.
3.  The Notebook will auto-unzip and parse it.
