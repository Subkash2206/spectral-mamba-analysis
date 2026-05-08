# SpectralMamba: Unified Spectral Audit Report

## 1. Executive Summary: Unmasking Spectral Artifacts
This report summarizes the corrected spectral findings for VM-UNet, Swin-UNet, and ResNet-UNet. Following the discovery that uncentered 2D-FFTs introduced significant DC-component bias, we re-executed our spectral audit using **mean-centered feature maps**. 

The results fundamentally shift our understanding: the "Spectral Debt" (aliasing) in Mamba is a **stage-specific architectural trait** rather than a global predictor of performance. The previously observed correlation between global AVR and Boundary F1 (BF1) has been proven to be an artifact of evaluation methodology.

---

## 2. Unified Spectral Audit Results (Mean-Centered AVR)

Results captured at matching resolution levels across all architectures using the unified `avr_stagewise_all.py` pipeline.

| Architecture | Level 1 (~64x64) | Level 2 (~32x32) | Level 3 (~16x16) | Level 4 (~8x8) | Mean AVR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UNet-ResNet50** | 0.3419 | 0.3587 | 0.2960 | 0.1804 | 0.2943 |
| **Swin-Tiny** | 0.3291 | 0.3671 | 0.2493 | 0.3646 | 0.3275 |
| **VM-UNet (Mamba)** | **0.4391** | **0.4012** | 0.1563 | 0.1370 | **0.2834** |

---

## 3. Findings & Conclusions

### I. The Correlation Artifact
The most significant finding of this audit is the **collapse of the AVR-BF1 correlation**. 
*   **Old Finding**: High aliasing directly causes boundary precision deficits (Pearson *r* ≈ -0.50).
*   **New Finding**: Once mean-centered, the pooled correlation is **+0.0541 (p=0.51)**, and the partial correlation (controlling for model identity) is **+0.0004 (p=0.99)**.
*   **Conclusion**: Global spectral aliasing does not explain the boundary performance gap between architectures. The previous "link" was an artifact of activation magnitudes (DC component) being captured by the FFT.

### II. Mamba’s Dual-Stage Spectral Behavior
While the global correlation is gone, Mamba exhibits a unique spectral trajectory:
1.  **Early Spectral Debt**: At Level 1, Mamba has the highest aliasing (AVR 0.44), with 21.8% of energy in the high-frequency band (vs 13% for CNN). This indicates significant high-frequency leakage in the initial encoding layers.
2.  **Aggressive Deep Filtering**: By Level 4 (the bottleneck), Mamba becomes the most aggressive low-pass filter (AVR 0.14), with only 3.7% high-frequency energy remaining. This suggests the Selective Scan mechanism performs a severe spectral compression in deep layers.

### III. Architectural Implications
Mamba's boundary precision (BF1 0.31) is superior to ResNet-UNet (0.26) but trails Swin-Tiny (0.37). This study confirms that this ranking is **not** a simple function of spectral aliasing. Mamba’s strength lies in its global context, while its Level 1 "spectral noise" may be a byproduct of its unique recurrent-like processing rather than a fatal flaw for edge detection.

---

## 4. Methodology & Reproducibility
*   **Correction**: All FFTs computed after `x = x - x.mean(dim=(-2, -1), keepdim=True)`.
*   **Weights**: Loaded with `strict=True` to ensure no architectural mismatch.
*   **Script**: `per_image_correlation.py` and `avr_stagewise_all.py`.
*   **Data Source**: `results/avr_stagewise_results_matched.csv` and `correlation_results.csv`.
