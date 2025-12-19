# Speech Spectrum Capture Tool

Based on the working Phase 3 EMG capture tool. **ONLY THE LABELS ARE CHANGED** - core serial/capture code is identical.

## Quick Start

```bash
chmod +x run_capture.sh
./run_capture.sh
```

Select:
- **Option 1**: Mouthing (Level 3) → Training data
- **Option 2**: Subvocal (Level 4) → Testing data

## The Protocol (from docs)

1. **Mouthing First** (Training): Open-mouth, exaggerate tongue movements
2. **Subvocal Second** (Testing): Same movements, but lips closed

## Labels Used

| Label | Word | Why It Works |
|-------|------|--------------|
| `GHOST` | "GHOST" | G slams tongue against palate = huge signal |
| `LEFT` | "LEFT" | L activates tongue tip strongly |
| `STOP` | "STOP" | Plosives engage jaw |
| `REST` | (silence) | Control - tongue flat, relaxed |

## Output Files

- `mouthing_data.csv` - Level 3 training data
- `subvocal_data.csv` - Level 4 testing data

Format: `Label,Timestamp,RawValue`
