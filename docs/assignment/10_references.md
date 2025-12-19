# Section 10: References

## Academic Papers

### Core EMG & Signal Processing

1. **Raez, M. B. I., Hussain, M. S., & Mohd-Yasin, F. (2006).** "Techniques of EMG signal analysis: detection, processing, classification and applications." *Biological Procedures Online*, 8, 11–35.

2. **Hopkins, J. (n.d.).** "Electromyography (EMG)." *Johns Hopkins Medicine*. [Link](https://www.hopkinsmedicine.org/health/treatment-tests-and-therapies/electromyography-emg)

### AlterEgo & Silent Speech

3. **Kapur, A., Kapur, S., & Maes, P. (2018).** "AlterEgo: A Personalized Wearable Silent Speech Interface." *Proceedings of the 23rd Int. Conf. on Intelligent User Interfaces (IUI)*, 43–53.

4. **Nieto, N., et al. (2022).** "Inner speech recognition through EEG." *arXiv preprint*.

### Machine Learning Architectures

5. **Breiman, L. (2001).** "Random Forests." *Machine Learning*, 45(1), 5–32.

6. **Chen, T., & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *KDD 2016*.

7. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** "Deep Residual Learning for Image Recognition." *CVPR*, 770–778.

8. **Sandler, M., et al. (2018).** "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR*.

9. **Hochreiter, S., & Schmidhuber, J. (1997).** "Long Short-Term Memory." *Neural Computation*, 9(8), 1735–1780.

10. **Vaswani, A., et al. (2017).** "Attention Is All You Need." *NeurIPS*.

11. **Szegedy, C., et al. (2015).** "Going Deeper with Convolutions." *CVPR*.

### Hardware Documentation

12. **Analog Devices. (2013).** "AD8232: Single-Lead Heart Rate Monitor Front End." [Datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/ad8232.pdf)

13. **Espressif Systems. (2020).** "ESP32 Technical Reference Manual." [Link](https://www.espressif.com/sites/default/files/documentation/esp32_technical_reference_manual_en.pdf)

### Phase 3 Prior Work

14. **Kho, C. V. L. (2025).** "Pareto-Optimal Model Selection for Low-Cost, Single-Lead EMG Control in Embedded Systems." [GitHub](https://github.com/CarlKho-Minerva/v2-emg-muscle)

## Code Repositories

- **Phase 3 Repository:** https://github.com/CarlKho-Minerva/v2-emg-muscle
- **scikit-learn:** https://scikit-learn.org/
- **TensorFlow/Keras:** https://www.tensorflow.org/
- **librosa (Audio Processing):** https://librosa.org/

## Datasets

- **Phase 4 Dataset:** Single-channel Silent Articulation sEMG
  - **Total samples:** 1,221,527
  - **Training (L3 Mouthing):** 515,547 samples → 200 windows
  - **Testing (L4 Subvocal):** 537,901 samples → 201 windows
  - **Colab Notebook:** [Google Colab](https://colab.research.google.com/drive/1gs-ES2spTU45gKnunJ4CoxcrCUpRzlxG)

## Key Findings Summary

| Metric | Phase 3 (Forearm) | Phase 4 (Subvocal) |
|--------|-------------------|-------------------|
| Best Multi-class | 74% (RF) | 24% (failed) |
| Binary Attempt | - | 72.64% (mode collapse) |
| Conclusion | Viable | **Not viable** |

**AI Statement**: I have used Gemini 3.0 Pro and Claude Opus 4.5 (Thinking) via the Antigravity IDE to speed up the process of data collection and analysis. All interpretations and high-level decisions/analysis are done by me.