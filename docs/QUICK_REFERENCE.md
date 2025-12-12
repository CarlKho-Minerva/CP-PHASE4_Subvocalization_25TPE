# PHASE 4: Subvocalization Quick Reference Card

## 🚀 The 3-Minute Pitch

MIT's AlterEgo costs **$1,200+**. This does the same thing for **$30**.

The **AD8232** (a $12 heart sensor) has a hardware filter that *accidentally* matches the exact frequency range needed for silent speech detection.

---

## 🔑 Key Numbers

| Metric | Value |
|--------|-------|
| Target Frequency | 1.3-50 Hz (AlterEgo) |
| AD8232 Bandpass | 0.5-40 Hz ✅ |
| Sample Rate | 1000 Hz |
| Channels | 2 (chin + jaw) |
| Total Cost | ~$30 USD |

---

## 📍 Electrode Placement (ASCII Art)

```
         👂 Green (Ground)
          │ (behind ear - Mastoid)
          │
    ┌─────┴─────┐
    │   HEAD    │
    └─────┬─────┘
          │
    🔴────┼────🟡  ← CH1: Under chin
     (2-3cm apart)     (Digastric muscle)
          │
    🔴────┼────🟡  ← CH2: Jaw cheek
          │           (Masseter muscle)
          ▼
```

---

## 🗣️ Best Words to Detect

| Word | Why It Works |
|------|--------------|
| **GHOST** | "G" slams tongue against palate = huge signal |
| **LEFT** | "L" activates tongue tip strongly |
| **STOP** | Plosives engage jaw |

**Control Word:** "MAMA" (lips only, no tongue → should see nothing)

---

## 📈 Training Protocol

1. **Whisper** → Get baseline signal
2. **Mouth/Mime** → Exaggerate silently (TRAIN HERE)
3. **Subvocalize** → Mouth closed (TEST HERE)
4. **Fade** → Reduce effort gradually

Key insight: ZCR (frequency) stays similar even when amplitude drops!

---

## ⚠️ Top 3 Failure Modes

1. **Floating SDN pin** → Wire to 3.3V!
2. **Wall power** → Use battery only!
3. **Long cables** → Keep <20cm, twist/shield!

---

## 🔧 Sanity Check Sequence

```
1. Heartbeat test (chest) → See spikes every second?
2. Jaw clench test (cheek) → Bite = explosion?
3. Subvocal test (chin) → "GHOST" = wiggle?
```

If all pass, you're ready to collect real data.

---

## 🧠 The Big Insight

> "You are building a **Tongue Tracker**, not a Mind Reader."

It's not telepathy. It's intercepting the firing order to muscles before sound is produced.

Level 3 (Mouthing) trains the model on big signals.
Level 4 (Subvocal) tests on tiny signals.
Transfer learning bridges the gap.
