# Dual AD8232 Troubleshooting Session
**Date:** December 19, 2024
**Video Documentation:** [Loom Recording](https://www.loom.com/share/a893fc0e55334356979a57ffecdbcfa3)

## Issue Summary

Two AD8232 sensors showing significantly different baseline ADC values:
- **Sensor 1:** ~1.8K baseline, Heart LED flickering ✅
- **Sensor 2:** ~3.8K baseline, Heart LED NOT flickering ⚠️

## Initial Concern

The second AD8232 was suspected to be broken because:
1. No Heart LED flicker
2. ADC baseline near upper limit (~3.8K approaching 4095 ceiling)
3. Risk of signal saturation on any muscle clench

## Diagnostic Testing

### Independent ESP32 Test
Tested the "broken" AD8232 on a separate NodeMCU-32S:

| Test Condition | Result | Interpretation |
|----------------|--------|----------------|
| No electrodes connected | "Board responding (leads off)" | Lead-off detection works ✅ |
| Touching all leads | "ADC RAILING HIGH" | Amplifier saturates (expected) |

**Conclusion:** Board is NOT dead - just has different baseline characteristics.

## Serial Monitor Output (from video)

```
ADC: 3796 | LO+: 1 | LO-: 1 | Status: ✓ Board responding
ADC: 3823 | LO+: 1 | LO-: 1 | Status: ✓ Board responding
ADC: 3797 | LO+: 1 | LO-: 1 | Status: ✓ Board responding
...
ADC: 3921 | LO+: 1 | LO-: 1 | Status: ⚠️ ADC RAILING HIGH - Possible SDN issue or dead IC
```

When holding electrodes tightly, ADC spikes above 3900 threshold → triggers warning message.

## Solution: Per-Sensor Baseline Calibration

Instead of dividing by 2 or matching baselines manually, implement **runtime calibration**:

1. On startup, read 200 samples from each sensor
2. Calculate individual baseline per sensor
3. Output **deviation from baseline** (centered around 0)
4. Both channels become directly comparable

### Firmware Created
`firmware/dual_channel_calibrated/dual_channel_calibrated.ino`

## Key Learnings

1. **Different AD8232 boards can have different DC offsets** - this is normal manufacturing variance
2. **Heart LED flicker ≠ board functionality** - LED responds to processed heartbeat, not raw signal
3. **Baseline calibration is essential** for multi-sensor setups
4. **ADC saturation risk** when baseline is near rails - calibration + deviation approach mitigates this

## Next Steps

- [ ] Test dual-channel calibrated firmware with electrodes attached
- [ ] Verify both channels produce comparable deviation signals
- [ ] Consider shaving for better electrode contact 😅

---

*"Might need to shave though." - CVK, 2025*
