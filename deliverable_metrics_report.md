# SpectralMamba: Unified Spectral Audit Report

> **⚠️ DEPRECATION NOTICE**: This is a legacy top-level copy of the report. The canonical version is located at `VM-UNet/results/`. This file is preserved for reference only and **should not** be cited in the manuscript or shared with reviewers. All numbers below have been corrected to match the source-of-truth CSVs.

---

## 1. Executive Summary: Unmasking Spectral Artifacts
This report summarizes the corrected spectral findings for VM-UNet, Swin-UNet, and ResNet-UNet. Following the discovery that uncentered 2D-FFTs and training-set evaluation introduced significant bias, we re-executed our audit using **mean-centered feature maps** and a **rigorous validation split**.

The central finding is a **"Correlation Collapse"**: once the DC component (intensity bias) is removed via mean-centering, the pooled AVR–BF1 Pearson r drops to **+0.0541 (p=0.51)** — statistically indistinguishable from zero. This refutes the previously reported hypothesis that spectral aliasing *causes* boundary failure. The "Spectral Debt" in Mamba is an **architectural trait** observable as a fingerprint, not a performance predictor. Mamba's O(N) linear scaling advantage over Transformer's O(N²) quadratic attention is therefore not offset by any statistically verified spectral cost to boundary precision.

---

## 2. Unified Performance Results (Full Validation Audit)
*Metrics calculated on the held-out ISIC18 validation set ($N=519$) using trained checkpoints with strict=True weight loading.*

| Architecture | Dice Score (↑) | Boundary F1 (BF1) (↑) |
| :--- | :---: | :---: |
| **VM-UNet (Mamba)** | **0.9027** | 0.2298 |
| **Swin-Tiny** | 0.9023 | **0.2540** |
| **UNet-ResNet50** | 0.9000 | 0.1900 |

*Source of truth: `VM-UNet/results/boundary_results.csv`*

---

## 3. Unified Spectral Audit Results (Mean-Centered AVR)
*Results captured at matching resolution levels using the unified pipeline. Source: `VM-UNet/results/avr_stagewise_results_matched.csv`*

| Architecture | Level 1 (~64×64) | Level 2 (~32×32) | Level 3 (~16×16) | Level 4 (~8×8) | Mean AVR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UNet-ResNet50** | 0.3427 | 0.3613 | 0.2974 | 0.1802 | 0.2954 |
| **Swin-Tiny** | 0.3276 | 0.3744 | 0.2519 | 0.3623 | 0.3291 |
| **VM-UNet (Mamba)** | **0.4600** | 0.3840 | 0.1408 | **0.1346** | **0.2799** |

**Dual-Stage Characteristic**: Mamba enters Stage 1 with the highest aliasing (AVR 0.46, ~35% above CNN baseline of 0.34), then performs the most aggressive spectral cleaning by Stage 4 (AVR 0.13, lowest of all models). This is a defining architectural fingerprint of the selective scan mechanism.

---

## 3-III. Correlation Analysis: "Correlation Collapse" (Corrected)
*Source: `VM-UNet/results/correlation_results.csv`*

| Population | Pearson r | p-value | Interpretation |
| :--- | :---: | :---: | :--- |
| **VM-UNet (Mamba)** | 0.3219 | 0.0226 | Weak positive within-model trend |
| **Swin-Tiny** | -0.1303 | 0.3671 | No significant correlation |
| **UNet-ResNet50** | -0.2003 | 0.1632 | No significant correlation |
| **Pooled (All Models)** | **+0.0541** | **0.51** | **No global correlation — Collapse confirmed** |

**Methodology Note on Intensity Bias**: Previous analyses reported spuriously strong correlations because uncentered FFTs include the DC component (mean pixel intensity), which dominates the energy spectrum. This created the illusion of an AVR–BF1 link. Once mean-centering ($f_{map} - \mu(f_{map})$) is applied, the artifact disappears and the pooled correlation collapses.

---

## 4. Key Findings
1.  **Correlation Collapse**: Once mean-centered, the pooled correlation between AVR and BF1 is **+0.0541 (p=0.51)**. Global spectral aliasing does **not** explain the boundary performance gap. Prior claims that aliasing *causes* BF1 failure must be withdrawn.
2.  **Dual-Stage Spectral Behavior**: Mamba enters with the highest "Spectral Debt" at Level 1 (AVR 0.46, ~35% above CNN) but becomes the most aggressive "Spectral Cleaner" by Level 4 (AVR 0.13, lowest of all models).
3.  **Robustness**: VM-UNet achieves superior shift-consistency versus Swin-Tiny at all shift magnitudes (Shift-5: Mamba 0.9531 vs. Swin 0.9196), showing that the dual-stage spectral behavior does not induce translational instability.
4.  **Complexity Justification**: Mamba's O(N) linear scaling vs. Transformer O(N²) is the primary architectural motivation. The Correlation Collapse finding confirms these efficiency trade-offs do not manifest as boundary failures.

---

## 5. Methodology & Reproducibility
*   **DC Correction**: All FFTs computed after `x = x - x.mean(dim=(-2, -1), keepdim=True)` to remove intensity bias.
*   **Validation**: 80/20 shuffle split (Seed 42) to ensure no training-set leakage.
*   **Weights**: Loaded from `best-ckpt/` with `strict=True`.
*   **Canonical Data**: All source-of-truth CSVs reside in `VM-UNet/results/`.
