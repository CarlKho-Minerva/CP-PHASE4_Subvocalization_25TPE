# GPIO Pin Mismatch Fix - AD8232 Channel 2

**Date:** December 18, 2024  
**Issue:** Channel 2 (AD8232 #2) showing constant value ~1495 instead of varying EMG signal

---

## Problem Diagnosis

### Symptoms
- AD8232 #1 (Channel 1 - Chin sensor): Working correctly, showing varying values ✅
- AD8232 #2 (Channel 2 - Jaw sensor): **Always reading ~1495** (constant/stuck) ❌

### Root Cause
**GPIO Pin Mismatch** between firmware code and physical wiring:

| Location | AD8232 #2 Pin Assignment | Correct? |
|----------|-------------------------|----------|
| **Firmware** (`subvocal_capture.ino` line 8) | `GPIO 35` | ❌ WRONG |
| **Wiring Guide** (`breadboard_wiring_guide.md` line 136) | `GPIO 36 (VP)` | ✅ CORRECT |

### Why This Causes the Issue
- The physical wire connects AD8232 #2 OUTPUT to **GPIO36** (as per wiring guide)
- The firmware reads from **GPIO35** (unconnected pin)
- GPIO35 is **floating** (no connection), so it reads random/constant voltage
- The value ~1495 represents approximately **36% of the 12-bit ADC range** (1495/4095 = 0.365)
- This is a typical mid-range floating pin reading

---

## The Fix

### Code Changes
Changed `firmware/subvocal_capture.ino`:

```diff
- #define AD8232_CH2    35    // Jaw (Masseter) - Secondary
+ #define AD8232_CH2    36    // Jaw (Masseter) - Secondary (VP pin)
```

Also updated the wiring reference comment:
```diff
- GPIO 35 ────────  OUTPUT
+ GPIO 36 ────────  OUTPUT
```

### Why GPIO36 (VP)?
- GPIO36 is also known as **VP (Voltage Positive)** pin
- It's an **ADC1_CH0** pin - one of the best analog input pins on ESP32
- Input-only pin with low noise characteristics
- Perfect for reading analog signals from AD8232

---

## Verification Checklist

After uploading the fixed firmware, verify:

- [ ] Connect electrodes to jaw/masseter muscle
- [ ] Open Serial Monitor at 115200 baud
- [ ] Observe both ch1 and ch2 values
- [ ] Clench jaw - ch2 should **spike significantly**
- [ ] Relax jaw - ch2 should return to **baseline (~1800-2000)**
- [ ] Values should **vary continuously**, not stay constant

### Expected Behavior
- **Before fix:** ch2 shows constant ~1495
- **After fix:** ch2 shows varying values that respond to jaw movement

---

## Technical Details

### ESP32 ADC Channels Used
| Channel | GPIO Pin | AD8232 Sensor | Body Location |
|---------|----------|---------------|---------------|
| CH1 | GPIO34 (ADC1_CH6) | AD8232 #1 | Chin (Digastric) |
| CH2 | **GPIO36 (ADC1_CH0)** | AD8232 #2 | Jaw (Masseter) |

### ADC Configuration
- Resolution: 12-bit (0-4095)
- Attenuation: 11dB (full 0-3.3V range)
- Sample Rate: 1000 Hz
- Communication: Serial (115200 baud)

---

## Related Files
- `firmware/subvocal_capture.ino` - Fixed firmware code
- `docs/working_process/2024-12-18_breadboard_wiring_guide.md` - Physical wiring reference (correct)

---

## Lesson Learned
**Always cross-reference pin assignments between:**
1. Firmware code definitions
2. Physical wiring documentation
3. Hardware setup photos/diagrams

A simple pin number typo can cause hours of debugging!
