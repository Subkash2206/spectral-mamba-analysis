# SpectralMamba: Unified Spectral Audit Report

## 1. Executive Summary: Unmasking Spectral Artifacts
This report summarizes the corrected spectral findings for VM-UNet, Swin-UNet, and ResNet-UNet. Following the discovery that uncentered 2D-FFTs and training-set evaluation introduced significant bias, we re-executed our audit using **mean-centered feature maps** and a **rigorous validation split**.

The results confirm that the "Spectral Debt" in Mamba is an **architectural trait** rather than a predictor of performance. While Mamba exhibits high aliasing in early layers, its selective scan mechanism performs aggressive low-pass filtering in deeper stages, achieving state-of-the-art Dice scores (~0.90) on the ISIC18 dataset.

---

## 2. Unified Performance Results (Full Validation Audit)
*Metrics calculated on the held-out ISIC18 validation set ($N=519$) using trained checkpoints.*

| Architecture | Dice Score (↑) | Boundary F1 (BF1) (↑) |
| :--- | :---: | :---: |
| **VM-UNet (Mamba)** | **0.9018** | 0.2291 |
| **Swin-Tiny** | 0.8976 | **0.2520** |
| **UNet-ResNet50** | 0.8982 | 0.1722 |

---

## 3. Unified Spectral Audit Results (Mean-Centered AVR)
*Results captured at matching resolution levels using the unified pipeline.*

| Architecture | Level 1 (~64x64) | Level 2 (~32x32) | Level 3 (~16x16) | Level 4 (~8x8) | Mean AVR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UNet-ResNet50** | 0.3419 | 0.3587 | 0.2960 | 0.1804 | 0.2943 |
| **Swin-Tiny** | 0.3291 | 0.3671 | 0.2493 | 0.3646 | 0.3275 |
| **VM-UNet (Mamba)** | **0.4628** | **0.3812** | 0.1411 | 0.1307 | **0.2789** |

---

## 4. Key Findings
1.  **Correlation Collapse**: Once mean-centered, the pooled correlation between AVR and BF1 is **+0.0541 (p=0.51)**. Global spectral aliasing does not explain the boundary performance gap.
2.  **Mamba’s Dual-Stage Behavior**: Mamba has the highest aliasing at Level 1 (AVR 0.46) but becomes the most aggressive filter by Level 4 (AVR 0.13).
3.  **Robustness**: VM-UNet shows superior shift-consistency compared to Swin-Tiny, losing only 2.2% IoU under pixel translations.

---

## 5. Methodology & Reproducibility
*   **Correction**: All FFTs computed after `x = x - x.mean(dim=(-2, -1), keepdim=True)`.
*   **Validation**: 80/20 shuffle split (Seed 42) to ensure no training-set leakage.
*   **Weights**: Loaded from `best-ckpt/` with `strict=True`.
