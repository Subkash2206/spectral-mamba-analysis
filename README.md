# Spectral Mamba: Unmasking Spectral Artifacts in Medical Image Segmentation

This project conducts a high-precision, authenticated audit of VM-UNet (Visual Mamba) to investigate the **"Spectral Debt"** hypothesis. We explore how the Selective Scan mechanism in State Space Models (SSMs) introduces architectural aliasing artifacts, and we rigorously test whether these artifacts statistically explain boundary segmentation deficits. Our key finding is a **"Correlation Collapse"**: once the DC component (intensity bias) is removed via mean-centering, the global link between spectral aliasing and boundary failure disappears entirely.

## TL;DR: Visual Summary of Spectral Debt

This research demonstrates that VM-UNet (Mamba) architectures exhibit a distinctive dual-stage spectral fingerprint — high aliasing at early layers, aggressive self-correction at deep layers — that is **not** causally linked to boundary segmentation failure once intensity bias is controlled.

| **1. Frequency Aliasing** | **2. Spectral Fingerprints** | **3. Correlation Collapse** |
| :---: | :---: | :---: |
| ![Band Decomposition](results/figures/band_decomposition.png) | ![Power Spectrum](results/figures/power_spectrum_grid.png) | ![Correlation Scatter](results/figures/avr_bf1_scatter.png) |
| High-frequency retention in early stages; aggressive filtering in deep stages. | Artifacts exposed via mean-centered 2D-FFT, isolating noise from intensity. | Pooled AVR–BF1 correlation collapses to r=+0.054 (p=0.51) after DC removal. |

---

## 1. Global Performance Audit (N=519)
Authenticated performance metrics on the full ISIC2018 validation set using the **strict=True** loading protocol.

| Architecture | Mean Dice Score | Boundary F1 (BF1) | Mean AVR (Spectral) | Audit Status |
| :--- | :---: | :---: | :---: | :--- |
| **VM-UNet (Mamba)** | **0.9027** | 0.2298 | **0.2799** | Authenticated |
| **Swin-Tiny** | 0.9023 | **0.2540** | 0.3291 | Authenticated |
| **UNet-ResNet50** | 0.9000 | 0.1900 | 0.2954 | Authenticated |

**Technical Significance**: All three architectures achieve near-identical semantic accuracy (Dice ~0.90). Mamba's O(N) linear scaling — versus Transformer's O(N²) quadratic self-attention — makes it the most computationally efficient architecture at scale. The BF1 differences among models are not explained by spectral aliasing (see Section 3), indicating that boundary precision is governed by other inductive biases rather than SSM scan artifacts alone.

---

## 2. Stage-wise Alias Volume Ratio (AVR)
The AVR measures the proportion of feature map energy in the high-frequency spectrum ($>0.5$ relative frequency), computed on **mean-centered** feature maps ($f_{map} - \mu(f_{map})$) to exclude DC-component intensity bias.

| Model | Stage 1 (64×64) | Stage 2 (32×32) | Stage 3 (16×16) | Stage 4 (8×8) | Mean AVR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UNet-ResNet50** | 0.3427 | 0.3613 | 0.2974 | 0.1802 | 0.2954 |
| **Swin-Tiny** | 0.3276 | 0.3744 | 0.2519 | 0.3623 | 0.3291 |
| **VM-UNet (Mamba)** | **0.4600** | 0.3840 | 0.1408 | 0.1346 | 0.2799 |

### Visualization: Stage-wise AVR Comparison
![Stagewise AVR Bars](results/figures/stagewise_avr_bars.png)
**Explanation**: This chart reveals Mamba's **Dual-Stage Characteristic**. At Stage 1 (highest resolution, 64×64), Mamba enters with a "Spectral Debt" of AVR 0.46 — approximately **35% higher** than the CNN baseline (0.34). However, by Stage 4 (8×8), Mamba acts as the most aggressive **"Spectral Cleaner"** among all architectures, reaching the lowest AVR of 0.13. This front-loaded debt followed by deep-layer self-correction is a defining fingerprint of the selective scan mechanism.

---

## 3. Correlation Statistics: The "Correlation Collapse" Finding
Per-image Pearson correlation between spectral aliasing (AVR) and boundary segmentation performance (BF1), computed after correcting for **Intensity Bias** via mean-centering.

| Population | Pearson r | p-value | Interpretation |
| :--- | :---: | :---: | :--- |
| **VM-UNet (Mamba)** | 0.3219 | 0.0226 | Weak positive within-model trend |
| **Swin-Tiny** | -0.1303 | 0.3671 | No significant correlation |
| **UNet-ResNet50** | -0.2003 | 0.1632 | No significant correlation |
| **Pooled (All Models)** | **+0.0541** | **0.51** | **No global correlation — Collapse confirmed** |

### Visualization: Correlation Scatter and Regression
![Correlation Scatter](results/figures/avr_bf1_scatter.png)
**Explanation**: This is the **"Correlation Collapse"** — the central finding of this audit. Previously reported strong correlations between AVR and BF1 were artifacts of **Intensity Bias**: the DC component of uncentered FFTs dominated the AVR signal, creating spurious correlations with overall image brightness rather than true architectural aliasing. Once mean-centered FFTs are used, the pooled Pearson r collapses to **+0.0541 (p=0.51)** — statistically indistinguishable from zero. This refutes the earlier hypothesis that spectral aliasing *causes* boundary failure. Mamba's O(N) scalability advantage therefore comes without a statistically verifiable spectral cost to boundary precision.

---

## 4. Translation Equivariance (Shift Consistency)
Mean IoU consistency between original predictions and predictions from sub-pixel shifted inputs.

| Model | Shift 1 | Shift 2 | Shift 3 | Shift 4 | Shift 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UNet-ResNet50** | **0.9864** | **0.9759** | **0.9678** | **0.9708** | **0.9693** |
| **Swin-Tiny** | 0.9670 | 0.9531 | 0.9484 | 0.9504 | 0.9196 |
| **VM-UNet (Mamba)** | 0.9755 | 0.9615 | 0.9551 | 0.9580 | **0.9532** |

### Visualization: Shift Consistency Curves
![Shift Consistency Curves](results/figures/shift_consistency_curves.png)
**Explanation**: VM-UNet maintains a Shift-5 consistency of **0.9532**, outperforming Swin-Tiny (**0.9196**) across all shift magnitudes. This demonstrates that Mamba's selective scan mechanism provides superior translation equivariance compared to window-based attention, while remaining competitive with CNNs. The spectral "Dual-Stage" behavior does not translate into translational instability at inference time.

---

## 5. Frequency Domain Diagnostics

### Frequency Band Decomposition
Analysis of energy distribution across Low ($<0.25$), Mid ($0.25–0.75$), and High ($>0.75$) bands.
![Band Decomposition](results/figures/band_decomposition.png)
**Explanation**: Mamba retains a "heavy tail" of aliased energy in its early stages (Stage 1 AVR 0.46), then performs aggressive low-pass correction in deep layers (Stage 4 AVR 0.13). CNNs follow a more monotonic filtering trajectory. This Dual-Stage profile is unique to the selective scan mechanism and constitutes Mamba's spectral fingerprint.

### Spectral Fingerprints (FFT Power Grids)
Mean-centered 2D-FFT heatmaps exposing architectural periodic noise.
![Power Spectrum Grid](results/figures/power_spectrum_grid.png)
**Explanation**: The cross-shaped artifacts in the Mamba rows correspond to the four-directional selective scan mechanism. These are structural fingerprints of the SSM scan order. Critically, after mean-centering removes intensity bias, these artifacts show no statistically significant correlation with boundary degradation (pooled r=+0.054, p=0.51).

---

## Methodology and Audit Rigor

1. **Global Mean-Centering**: All spectral metrics utilize $f_{map} - \mu(f_{map})$ to isolate frequency noise from intensity bias (DC component). This correction was the key methodological step that resolved the "Intensity Bias" artifact in prior correlation analyses.
2. **Strict=True Protocol**: Enforced 100% state-dict matching using the `flexible_load` utility. This ensures every weight, including selective scan parameters, is authenticated.
3. **Boundary F1 Protocol**: Edge precision calculated using morphological erosion with a distance threshold $D=2$.
4. **Complexity Context**: Mamba's O(N) linear scaling versus Transformer's O(N²) quadratic self-attention is the primary motivation for investigating whether spectral trade-offs exist. The Correlation Collapse finding confirms these trade-offs do not manifest as boundary failures at ISIC18 scale.

## Reproduction Guide
```bash
python tools/boundary_eval.py    # Global Performance (N=519)
python run_band_only.py          # Band Decomposition & Shift Analysis
python tools/master_avr_audit.py # Stage-wise Spectral Audit
python VM-UNet/visualizations.py # Publication Figure Generation
```
