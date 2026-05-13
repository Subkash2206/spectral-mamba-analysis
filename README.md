# Spectral Mamba: Unmasking Spectral Artifacts in Mamba-Based Medical Image Segmentation


A rigorous spectral audit of VM-UNet (Visual Mamba) against Swin-Tiny and UNet-ResNet50 on dermatological image segmentation. We introduce and operationalize the **Alias Violation Ratio (AVR)**, a mean-centered, DC-corrected spectral aliasing metric applied directly to intermediate encoder feature maps, to test the *Spectral Debt* hypothesis: that the Selective Scan mechanism in State Space Models (SSMs) introduces architectural aliasing artifacts that explain boundary segmentation deficits.


The findings do not support the hypothesis that spectral aliasing globally explains boundary segmentation deficits. Instead, the analysis reveals a previously unreported dual-stage spectral fingerprint associated with the SSM scan order, whose existence is architectural and whose consequences for boundary precision become statistically negligible once intensity bias is properly removed.

---

## Table of Contents

1. [TL;DR](#tldr)
2. [Background and Motivation](#background-and-motivation)
3. [Global Performance Audit](#1-global-performance-audit)
4. [Stage-wise AVR: The Dual-Stage Fingerprint](#2-stage-wise-avr-the-dual-stage-fingerprint)
5. [Correlation Analysis: The Collapse](#3-correlation-analysis-the-collapse)
6. [Translation Equivariance](#4-translation-equivariance-shift-consistency)
7. [Frequency Domain Diagnostics](#5-frequency-domain-diagnostics)
8. [Methodology and Audit Rigor](#6-methodology-and-audit-rigor)
9. [Model Architectures and Training](#7-model-architectures-and-training)
10. [Repository Structure](#8-repository-structure)
11. [Reproduction Guide](#9-reproduction-guide)
12. [Known Limitations](#10-known-limitations)
13. [Citation](#citation)

---

## TL;DR

Three architectures, one dataset, one question: does Mamba's spectral aliasing cost it boundary precision?

| | Frequency Aliasing | Spectral Fingerprints | Correlation Collapse |
| :---: | :---: | :---: | :---: |
| | ![Band Decomposition](results/figures/band_decomposition.png) | ![Power Spectrum](results/figures/power_spectrum_grid.png) | ![AVR-BF1 Scatter](results/figures/avr_bf1_scatter.png) |
| **What it shows** | Mamba front-loads high-frequency energy at Stage 1, then aggressively self-corrects by Stage 4 | Mean-centered 2D-FFT heatmaps expose cross-shaped scan artifacts unique to the SSM scan order | Pooled AVR-BF1 Pearson *r* collapses to +0.0108 (*p* = 0.670) after DC removal |
| **What it means** | A dual-stage spectral fingerprint not shared by CNNs or Transformers | Structurally distinctive, not pathological | Spectral aliasing does not globally explain boundary failure |

Mamba's O(N) linear scaling advantage over Transformer's O(N²) self-attention comes without a statistically verifiable spectral cost to boundary precision. The observed aliasing patterns appear to be architectural characteristics of the SSM scan mechanism rather than pathological predictors of boundary failure.

---

## Background and Motivation

State Space Models, and Vision Mamba architectures in particular, have emerged as a compelling alternative to both CNNs and Vision Transformers for dense prediction tasks.VM-UNet achieves competitive segmentation accuracy with linear computational scaling — a substantial efficiency advantage over window-based Transformer attention mechanisms such as Swin-Tiny. However, the Selective Scan mechanism, which processes spatial tokens along four directional traversals (horizontal, vertical, and their reverses), has unknown spectral properties. Unlike convolutions, which have well-characterized frequency responses tied to filter support, or Transformers, whose global self-attention aggregates all spatial frequencies simultaneously, the sequential scan has no obvious frequency-domain prior.

**The proposed Spectral Debt hypothesis** posits that this scan order introduces boundary-frequency leakage — aliasing artifacts that are preferentially harmful to boundary-level precision (as measured by Boundary F1), even when semantic accuracy (Dice) is unaffected. This hypothesis is plausible: if Mamba's scan leaves high-frequency edge information poorly resolved at early stages, downstream boundary delineation could suffer systematically.

This work performs a controlled test of that hypothesis. The key methodological contribution is the DC correction step: all spectral analysis operates on **mean-centered feature maps**, which strips out the DC component (mean pixel intensity) that otherwise dominates FFT energy spectra and creates spurious correlation artifacts. Prior analyses that reported strong AVR-BF1 links (Pearson *r* ~ -0.50) failed to apply this correction, and this work demonstrates that the reported link was entirely an artifact of intensity bias rather than structural frequency-domain pathology.

---

## 1. Global Performance Audit

**N = 519** images from the ISIC2018 validation split (a fixed-seed 20% hold-out of the standard ISIC2018 training set; see Section 6 for split details). All checkpoints loaded with `strict=True` via the `flexible_load` utility — 100% state-dict authenticated, no partial weight loading.

| Architecture | Dice (↑) | BF1 (↑) | Mean AVR | Best Val Dice (Training) |
| :--- | :---: | :---: | :---: | :---: |
| **VM-UNet (Mamba)** | **0.9027** | 0.4939 | **0.2799** | 0.9163 |
| **Swin-Tiny** | 0.9023 | **0.5259** | 0.3291 | 0.9061 |
| **UNet-ResNet50** | 0.9000 | 0.4470 | 0.2954 | 0.9083 |

*Source: `VM-UNet/results/boundary_results.csv`*

All three architectures converge to near-identical Dice (~0.90), indicating that semantic segmentation accuracy saturates at ISIC2018 scale regardless of architectural inductive bias. The 2.7-point BF1 gap between Swin-Tiny and VM-UNet is real, but as Section 3 demonstrates, it is not attributable to spectral aliasing.

Mamba also carries the *lowest* mean AVR of the three architectures (0.2799 vs. 0.3291 for Swin and 0.2954 for UNet). This seems paradoxical given the Spectral Debt framing; the stage-wise breakdown in Section 2 resolves the apparent contradiction.

---

## 2. Stage-wise AVR: The Dual-Stage Fingerprint

**AVR (Alias Violation Ratio)** measures the proportion of feature map energy concentrated above the 0.5 relative frequency threshold associated with aliasing-sensitive regions of the frequency plane. All AVRs are computed on mean-centered feature maps to suppress the DC component.

$$\text{AVR}(f) = \frac{\displaystyle\sum_{\xi\,:\,|\xi|_\infty > 0.5} \bigl|\hat{f}(\xi)\bigr|^2}{\displaystyle\sum_{\xi} \bigl|\hat{f}(\xi)\bigr|^2}, \qquad \hat{f} = \mathcal{F}\!\left[f - \mu(f)\right]$$

where $|\xi|_\infty = \max(|\xi_y|, |\xi_x|)$ is the Chebyshev frequency norm, frequencies are normalized to $[-1, 1]$, and $\mu(f)$ is the spatial mean of the feature map computed over the $(H, W)$ dimensions.

| Model | Stage 1 (64x64) | Stage 2 (32x32) | Stage 3 (16x16) | Stage 4 (8x8) | Mean AVR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UNet-ResNet50** | 0.3427 | 0.3613 | 0.2974 | 0.1802 | 0.2954 |
| **Swin-Tiny** | 0.3276 | 0.3744 | 0.2519 | 0.3623 | 0.3291 |
| **VM-UNet (Mamba)** | **0.4600** | 0.3840 | 0.1408 | **0.1346** | 0.2799 |

*Source: `VM-UNet/results/avr_stagewise_results_matched.csv`*

![Stage-wise AVR](results/figures/stagewise_avr_bars.png)

**The Dual-Stage Characteristic.** Mamba enters Stage 1 at AVR 0.46 — approximately 35% above the CNN baseline of 0.34. This is the *Spectral Debt*: substantial high-frequency energy at full spatial resolution, a direct consequence of the four-directional Selective Scan processing tokens before downsampling. By Stage 4, however, Mamba exhibits the strongest high-frequency suppression of the three, reaching AVR 0.13 — well below Swin (0.36) and UNet (0.18). This front-loaded debt followed by deep-layer self-correction is a defining fingerprint of the SSM selective scan, not shared by CNNs (smooth monotonic decline) or Transformers (plateau behavior from Stages 2-4).

This trajectory explains why mean AVR is lower for Mamba than for Swin: the Stage 4 compression is sufficiently aggressive to pull the four-stage average below both baselines, despite Mamba's dominant Stage 1.

---

## 3. Correlation Analysis: The Collapse

Per-image Pearson correlation between mean AVR (averaged across all four encoder stages) and Boundary F1, computed across the full N = 519 validation set after DC correction.

| Population | Pearson *r* | *p*-value | N | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| VM-UNet (Mamba) | +0.0998 | 0.0229 | 519 | Weak but significant within-model trend |
| Swin-Tiny | +0.0188 | 0.6686 | 519 | No significant correlation |
| UNet-ResNet50 | -0.1880 | < 0.0001 | 519 | Weak but significant within-model trend |
| **Pooled (all models)** | **+0.0108** | **0.6704** | **1,557** | **No global correlation — Collapse confirmed** |
| **Partial (model-controlled)** | **-0.0001** | **0.9956** | **1,557** | **No structural link** |

*Source: `VM-UNet/results/correlation_results.csv`*

![AVR-BF1 Scatter](results/figures/avr_bf1_scatter.png)

**The Correlation Collapse.** Mamba shows a statistically significant within-model trend (*r* = 0.0998, *p* = 0.023), but this disappears when pooling across architectures: pooled *r* = +0.0108 (*p* = 0.670), statistically indistinguishable from zero. The partial correlation — computed by regressing both AVR and BF1 against a one-hot architecture indicator matrix and correlating the residuals — yields *r* = -0.0001 (*p* = 0.996), confirming that the within-model signals are architecture-specific phenomena, not a structural spectral-to-boundary link.

**What the within-model signals mean.** Mamba (*r* = +0.099) and UNet (*r* = -0.188) show opposite-sign within-model trends.If spectral aliasing were a shared, directional driver of boundary failure, within-model correlations would be expected to share sign. They do not, which is consistent with the interpretation that each architecture's within-model AVR-BF1 pattern reflects idiosyncratic image-difficulty confounds rather than a causal spectral mechanism.

**The intensity bias artifact.** The previous (erroneous) finding of strong AVR-BF1 correlation was an artifact of uncentered FFTs. Without mean-centering, the DC component dominates the energy spectrum — bright images accumulate high DC energy that simultaneously predicts both high denominator AVR and easier segmentation tasks. This creates a spurious negative correlation. The DC correction $f \leftarrow f - \mu(f)$ eliminates this confound entirely, and the correlation collapses.

---

## 4. Translation Equivariance (Shift Consistency)

Shift consistency measures whether a model produces stable predictions under small spatial translations — a proxy for translation equivariance at inference time. For each validation image, a horizontal pixel shift of magnitude $s$ is applied via `torch.roll`, inference is run on the shifted image, the output is shifted back by $-s$, and mean IoU is computed against the unshifted baseline prediction.

| Model | Shift 1 | Shift 2 | Shift 3 | Shift 4 | Shift 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UNet-ResNet50** | **0.9843** | **0.9773** | **0.9730** | **0.9730** | **0.9702** |
| **Swin-Tiny** | 0.9621 | 0.9520 | 0.9487 | 0.9483 | 0.9391 |
| **VM-UNet (Mamba)** | 0.9719 | 0.9616 | 0.9592 | 0.9624 | **0.9552** |

*Source: `VM-UNet/results/shift_consistency_results.csv`*

![Shift Consistency Curves](results/figures/shift_consistency_curves.png)

UNet-ResNet50 leads on shift consistency — expected, since strided convolutions with pooling provide approximate translation invariance by construction. More notable: **Mamba outperforms Swin-Tiny at every shift magnitude** (Shift-5 advantage: +1.61 IoU points, 0.9552 vs. 0.9391).

This is counterintuitive given Mamba's high Stage-1 AVR. Despite front-loading spectral energy into the high-frequency band, the Selective Scan mechanism does not propagate that instability through to output-level translation sensitivity. The dual-stage self-correction appears functionally effective: by the time activations reach the bottleneck, Mamba has suppressed the early spectral noise. Swin-Tiny's inferior shift consistency is consistent with the known sensitivity of window-based attention to boundary alignment between windows under spatial shifts.

**Note on protocol.** The `torch.roll` shift applies cyclic boundary conditions, introducing wrap-around pixels at the leading edge. For images with non-zero border content, this adds a small systematic artifact. All three models are evaluated under identical conditions, so relative comparisons are valid, but absolute shift consistency values should not be interpreted as ground-truth translation invariance metrics.

---

## 5. Frequency Domain Diagnostics

### 5a. Band Decomposition

Feature map energy partitioned into three frequency bands computed on mean-centered maps across all four encoder stages. Band boundaries are defined in normalized frequency units using the Chebyshev norm:

- **Low**: $|\xi|_\infty \leq 0.25$
- **Mid**: $0.25 < |\xi|_\infty \leq 0.75$
- **High**: $|\xi|_\infty > 0.75$

| Model | Stage | Low | Mid | High |
| :--- | :---: | :---: | :---: | :---: |
| UNet-ResNet50 | 1 | 0.394 | 0.480 | 0.126 |
| UNet-ResNet50 | 2 | 0.356 | 0.506 | 0.138 |
| UNet-ResNet50 | 3 | 0.430 | 0.465 | 0.105 |
| UNet-ResNet50 | 4 | 0.548 | 0.401 | 0.051 |
| Swin-Tiny | 1 | 0.558 | 0.283 | 0.159 |
| Swin-Tiny | 2 | 0.391 | 0.414 | 0.194 |
| Swin-Tiny | 3 | 0.440 | 0.466 | 0.094 |
| Swin-Tiny | 4 | ~0.000 | 0.891 | 0.109 |
| VM-UNet (Mamba) | 1 | 0.322 | 0.451 | **0.228** |
| VM-UNet (Mamba) | 2 | 0.372 | 0.460 | 0.168 |
| VM-UNet (Mamba) | 3 | 0.710 | 0.242 | 0.048 |
| VM-UNet (Mamba) | 4 | 0.694 | 0.274 | **0.033** |

*Source: `VM-UNet/results/band_decomposition_results.csv`*

![Band Decomposition](results/figures/band_decomposition.png)

At Stage 1, Mamba carries 22.8% of its energy in the high-frequency band — the highest of the three models. By Stage 4, this compresses to 3.3%, the lowest. The UNet trajectory is monotonically smooth. Swin exhibits an anomaly at Stage 4 where the low-band ratio collapses to machine epsilon (~2.7e-14): this is an expected artifact of LayerNorm in the final Swin transformer stage, which normalizes activations to zero mean by construction before the AVR measurement, effectively removing the DC component a second time and collapsing low-band energy. This does not affect AVR values (which are computed after an additional explicit mean-centering step) but makes absolute low-band ratios at Swin Stage 4 uninterpretable in isolation.

### 5b. Spectral Fingerprints (2D-FFT Power Grids)

Mean-centered 2D-FFT power heatmaps computed on the first validation image, shown per model per encoder stage, log-scale.

![Power Spectrum Grid](results/figures/power_spectrum_grid.png)

The cross-shaped artifacts in the Mamba rows are the clearest visual signature in this work. They correspond directly to the four-directional scan traversal of the SSM: horizontal and vertical sweeps and their reverses each contribute an oriented frequency signature, which manifests as a cross pattern in the 2D power spectrum. These artifacts are structural fingerprints of the scan mechanism — reproducible and interpretable — and are statistically inert with respect to boundary performance once intensity bias is removed (pooled *r* = +0.0108, *p* = 0.670).

---

## 6. Methodology and Audit Rigor

### 6a. AVR Definition

All spectral metrics use mean-centered feature maps to exclude the DC component:

$$\text{AVR}(f) = \frac{\displaystyle\sum_{\xi\,:\,|\xi|_\infty > 0.5} \bigl|\mathcal{F}[f - \mu(f)](\xi)\bigr|^2}{\displaystyle\sum_{\xi} \bigl|\mathcal{F}[f - \mu(f)](\xi)\bigr|^2}$$

The threshold 0.5 in the Chebyshev norm corresponds to the boundary between the inner quarter and outer three-quarters of the 2D frequency plane, consistent with standard antialiasing conventions at the relative Nyquist limit. The mean-centering $f \leftarrow f - \mu(f)$ is applied per feature map over the spatial $(H, W)$ dimensions using `fmap.mean(dim=(-2, -1), keepdim=True)`.

### 6b. Boundary F1 Protocol

Edge precision is evaluated via morphological boundary extraction with a pixel-tolerance distance threshold of $D = 2$:

1. Extract 1-pixel boundaries: `boundary = mask XOR binary_erosion(mask, iterations=1)`
2. Build signed distance transforms from both pred and GT boundaries via `distance_transform_edt(~boundary)`
3. Count true positives with tolerance: pred-boundary pixels within $D=2$ pixels of the GT boundary, and vice versa
4. Compute precision and recall from the tolerant TP counts
5. $\text{BF1} = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$

### 6c. Data Split

The ISIC2018 training set contains 2,594 dermoscopy images with corresponding binary lesion segmentation masks. A 20% fixed-seed hold-out of the training distribution was used for all spectral and performance audits (N = 519). The split is deterministic and reproducible:

```python
random.seed(42)
random.shuffle(sorted_image_paths)
val_imgs = sorted_image_paths[int(0.8 * len(sorted_image_paths)):]
```

Models were trained on the complementary 80% split (N = 2,075). All three models use the identical split, ensuring direct comparability. Note that this constitutes evaluation on a hold-out of the original training distribution rather than the official ISIC2018 test set; see Section 10 for implications.

### 6d. Feature Extraction Protocol

Intermediate features are extracted via PyTorch forward hooks registered on the following modules:

| Architecture | Hook Target | Rationale |
| :--- | :--- | :--- |
| UNet-ResNet50 | `encoder.layer1` through `encoder.layer4` | Output of each ResNet encoder stage |
| Swin-Tiny | `swin_unet.layers[i-1]` (i=1..4) | Output of each Swin transformer stage |
| VM-UNet (Mamba) | `vmunet.layers[i-1]` (i=1..4) | Output of each VSS (Visual State Space) stage |

For Swin and Mamba stages whose outputs are in channel-last format `(B, H, W, C)` or sequence format `(B, L, C)`, outputs are permuted or reshaped to `(B, C, H, W)` before AVR computation. Spatial dimensions at each stage are approximately 64x64, 32x32, 16x16, and 8x8 for a 256x256 input.

Note: `master_avr_audit.py` (authoritative AVR table) hooks full stage outputs, while `per_image_correlation.py` (authoritative correlation table) hooks the last block of each stage. These produce conceptually consistent but not numerically identical feature maps; the AVR values in Section 2 and the correlation analysis in Section 3 therefore use slightly different extraction points.

### 6e. Weight Authentication

All model checkpoints are loaded via the `flexible_load` utility with `strict=True`, which enforces 100% state-dict key matching after stripping `thop` profiling metadata (`total_ops`, `total_params` keys). The only permitted key transformation is `vmunet.` prefix normalization (added or removed as needed to match the model wrapper's attribute structure). No partial loading or weight interpolation is performed.

### 6f. Partial Correlation

The partial correlation (Table 3, last row) controls for architecture-level mean differences in both AVR and BF1 via OLS regression:

```python
X = one_hot_architecture_indicators  # shape (1557, 3)
resid_avr = avr - X @ np.linalg.lstsq(X, avr, rcond=None)[0]
resid_bf1 = bf1 - X @ np.linalg.lstsq(X, bf1, rcond=None)[0]
r_partial, p_partial = pearsonr(resid_avr, resid_bf1)
```

This isolates the within-architecture residual covariance from the pooled signal.

---

## 7. Model Architectures and Training

### Architecture Summary

| Model | Backbone | Input Size | Complexity | Pretrained Encoder |
| :--- | :--- | :---: | :--- | :--- |
| **VM-UNet** | VMamba (VSS blocks) | 256x256 | O(N) | No (scratch) |
| **Swin-Tiny** | Swin Transformer (patch4, window7) | 224x224 | O(N log N) | ImageNet-1K |
| **UNet-ResNet50** | ResNet50 | 256x256 | O(N) | ImageNet |

VM-UNet uses encoder depths `[2, 2, 9, 2]` and decoder depths `[2, 9, 2, 2]`, with drop path rate 0.2. Swin-Tiny uses depths `[2, 2, 2, 2]` with matching decoder depths `[1, 2, 2, 2]`, drop path rate 0.2. All models output a single-channel binary segmentation mask; VM-UNet applies sigmoid internally in its forward pass (`if self.num_classes == 1: return torch.sigmoid(logits)`) while UNet-ResNet50 and Swin-Tiny output raw logits.

### Training Protocol

All models trained for 100 epochs on the 80% ISIC2018 training split.

| Hyperparameter | VM-UNet | Swin-Tiny | UNet-ResNet50 |
| :--- | :---: | :---: | :---: |
| Optimizer | Adam | Adam | Adam |
| Learning rate | 1e-4 | 1e-4 | 1e-4 |
| Batch size | 4 | 8 | 8 |
| Loss function | Dice (`from_logits=False`) | Dice (`from_logits=True`) | Dice (`from_logits=True`) |
| Epochs | 100 | 100 | 100 |

Data augmentation (training only): horizontal flip (*p* = 0.5), vertical flip (*p* = 0.5), random rotation (*p* = 0.5, range [0, 360] degrees). Normalization uses ISIC2018-specific per-channel mean and standard deviation. No augmentation is applied at validation time.

Checkpoints are saved at best validation Dice. Best validation Dice achieved during training: VM-UNet 0.9163, UNet-ResNet50 0.9083, Swin-Tiny 0.9061.

---

## 8. Repository Structure

```
spectral-mamba-analysis/
├── models/
│   └── vmunet/
│       ├── vmunet.py              # VM-UNet wrapper (sigmoid output for binary segmentation)
│       └── vmamba.py              # VMamba backbone (VSS blocks, SSM scan implementation)
├── tools/
│   ├── boundary_eval.py           # Global Dice + BF1 evaluation  [AUTHORITATIVE]
│   ├── master_avr_audit.py        # Stage-wise AVR audit           [AUTHORITATIVE]
│   ├── avr_analysis.py            # Per-architecture AVR analysis
│   ├── avr_swin_baseline.py       # Swin-specific AVR baseline
│   ├── avr_unet_baseline.py       # UNet-specific AVR baseline
│   ├── hook_test.py               # Forward hook validation utility
│   ├── check_win_rate.py          # Architecture win-rate comparison
│   └── quick_compare.py           # Quick cross-model comparison
├── configs/
│   └── config_setting.py          # Training configuration (VM-UNet canonical)
├── datasets/
│   └── dataset.py                 # ISIC2018 dataset loader (NPY and PNG formats)
├── per_image_correlation.py       # AVR-BF1 Pearson correlation     [AUTHORITATIVE]
├── shift_consistency.py           # Translation equivariance audit  [AUTHORITATIVE]
├── run_band_only.py               # Band decomposition + figure generation
├── regen_scatter_only.py          # Scatter figure regeneration (no inference required)
├── avr_stagewise_all.py           # Unified stage-wise AVR pipeline (root version)
├── visualizations.py              # Publication figure generation
├── verify_bands.py                # Band decomposition sanity check
├── train_vmunet_isic18.py         # VM-UNet training script
├── train_swinunet_isic18.py       # Swin-Tiny training script
├── train_unet_isic18.py           # UNet-ResNet50 training script
├── engine.py / engine_synapse.py  # Training engine utilities
├── utils.py                       # Loss functions, metrics, dataset utilities
├── run_experiments.sh             # Full three-model training pipeline
├── results/
│   └── figures/                   # Publication figures (PDF + PNG, 300 DPI)
└── VM-UNet/
    ├── best-ckpt/                 # Trained checkpoints (not tracked by git)
    ├── data/isic18/               # ISIC2018 dataset (not tracked by git)
    └── results/
        ├── boundary_results.csv              # Authoritative Dice + BF1
        ├── correlation_results.csv           # Authoritative AVR-BF1 Pearson r
        ├── shift_consistency_results.csv     # Authoritative shift IoU
        ├── avr_stagewise_results_matched.csv # Authoritative stage-wise AVR
        └── band_decomposition_results.csv    # Band energy ratios with std
```

Scripts marked `[AUTHORITATIVE]` write the canonical result CSVs. All other scripts are for analysis, figure generation, or development validation. Do not use intermediate outputs from `run_band_only.py` or `regen_scatter_only.py` as source-of-truth metrics.

---

## 9. Reproduction Guide

### Prerequisites

- Python 3.8
- PyTorch >= 1.13 with CUDA (CPU inference is possible but slow for N=519)
- Trained checkpoints in `VM-UNet/best-ckpt/`
- ISIC2018 images in `VM-UNet/data/isic18/train/images/` and masks in `VM-UNet/data/isic18/train/masks/`

Install dependencies:

```bash
pip install torch torchvision segmentation-models-pytorch scipy matplotlib tqdm pillow
```

### Training (skip if using provided checkpoints)

```bash
# All three models sequentially
bash run_experiments.sh

# Or individually from the repository root
python train_unet_isic18.py
python train_swinunet_isic18.py
python train_vmunet_isic18.py
```

### Generating Results

Run the following from the repository root in order:

```bash
# Step 1: Global performance — Dice + BF1 on N=519 validation set
# Writes: VM-UNet/results/boundary_results.csv
python tools/boundary_eval.py

# Step 2: Per-image AVR-BF1 correlation (central finding — Correlation Collapse)
# Writes: VM-UNet/results/correlation_results.csv
python per_image_correlation.py

# Step 3: Translation equivariance (shift consistency)
# Writes: VM-UNet/results/shift_consistency_results.csv
python shift_consistency.py

# Step 4: Stage-wise AVR (canonical hook-based pipeline)
# Writes: VM-UNet/results/avr_stagewise_results_matched.csv
python tools/master_avr_audit.py

# Step 5: Band decomposition figures and raw band CSV
# Writes: results/figures/band_decomposition.{png,pdf}
#         results/figures/power_spectrum_grid.{png,pdf}
#         results/figures/shift_consistency_curves.{png,pdf}
#         results/figures/stagewise_avr_bars.{png,pdf}
#         VM-UNet/results/band_decomposition_results.csv
python run_band_only.py

# Step 6: AVR-BF1 scatter figure (reads from correlation CSV, no model inference)
# Writes: results/figures/avr_bf1_scatter.{png,pdf}
python regen_scatter_only.py
```

All figures are saved as both PDF (vector, for paper submission) and PNG (300 DPI, for README and supplementary).

---

## 10. Known Limitations

**Validation split.** Evaluation is performed on a 20% hold-out of the ISIC2018 training distribution, not the official ISIC2018 test set. Results reflect in-distribution generalization within a fixed random split. Comparison to published ISIC2018 benchmarks that use the official test set should account for this difference.

**Single dataset.** All findings are derived from ISIC2018 dermoscopy images. Whether the dual-stage spectral fingerprint and correlation collapse generalize to other medical imaging modalities (CT, MRI, histopathology) or other Mamba-based architectures (MambaSeg, VM-UNet v2, Vim) is an open question.

**Horizontal-only shifts.** Shift consistency is measured under horizontal pixel shifts only. Vertical shifts, diagonal shifts, and sub-pixel shifts may yield different relative orderings; the current results should not be interpreted as a comprehensive translation equivariance characterization.

**Cyclic boundary conditions.** The `torch.roll` shift protocol introduces wrap-around pixels at image borders. Relative model rankings are reliable; absolute IoU values are slightly affected.

**Hook placement divergence.** The stage-wise AVR (Section 2, `master_avr_audit.py`) hooks full stage outputs, while the correlation analysis (Section 3, `per_image_correlation.py`) hooks the last block of each stage. These are conceptually consistent but not identical extraction points, which introduces a small methodological inconsistency between the two tables.

**Swin Stage 4 low-band anomaly.** The Swin-Tiny Stage 4 low-band energy ratio is machine epsilon (~2.7e-14) due to LayerNorm removing the spatial mean before measurement. This does not affect AVR values but makes absolute band ratio comparisons at Swin Stage 4 uninterpretable for the low-frequency component.

**Single-seed training.** All models were trained with a fixed seed but only once. Variance across training runs has not been quantified; Dice and BF1 values should be treated as point estimates. The spectral findings (AVR, correlation) are evaluated on fixed trained weights and are not subject to training variance.

---


## Acknowledgements

The VM-UNet architecture is adapted from [VM-UNet (Ruan et al., 2024)](https://github.com/JCruan519/VM-UNet). The Swin-UNet architecture is adapted from [Swin-Unet (Cao et al., 2022)](https://github.com/HuangBo-Terraloupe/SwinUnet). The UNet encoder-decoder framework uses the [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) library.


## License

This repository is released under the Apache License 2.0. See `LICENSE` for details.

The VM-UNet and Swin-UNet components retain their respective upstream licenses.
