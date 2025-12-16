# Section 9: Executive Summary

## Project Overview

**AlterEgo's Alter Ego**: Replicating MIT Media Lab's subvocalization detection for **$30** instead of **$1,200+**.

## Pipeline Diagram

```
Data → Preprocess → Features → Models → Deploy
 │         │           │         │        │
AD8232   Bandpass    Stats    MaxCRNN   ESP32
ESP32    Notch 60Hz  Raw Seq  RF        GPU
CSV      Windows     Spectro  CNN
```

## Key Results

| Model | Accuracy | Latency | Deployable? |
|-------|----------|---------|-------------|
| **MaxCRNN** | 83% | 0.15ms (GPU) | No |
| **Random Forest** | 74% | 0.01ms | **Yes** |

## Novel Contribution: MaxCRNN

**Inception + Bi-LSTM + Multi-Head Attention**
- 99% precision on safety-critical class
- Captures multi-scale temporal patterns

## Transfer Learning Strategy

Train on **Mouthing** (strong signals) → Test on **Subvocalization** (weak signals)

## Deployment Decision

- GPU available → MaxCRNN (83%)
- ESP32 only → Random Forest (74%, 0.01ms)

## Cost-Benefit

| System | Cost | Accuracy |
|--------|------|----------|
| MIT AlterEgo | $1,200+ | ~92% |
| **This Project** | **$30** | 74-83% |

*"Building a Biological Keyboard, not a Telepathy Helmet."*
