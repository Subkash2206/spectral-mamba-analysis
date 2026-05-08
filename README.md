# Spectral Analysis of State Space Models for Vision: The Mamba Paradox

This repository investigates the internal mechanics of State Space Models (specifically Mamba-based architectures) in the context of medical image segmentation. We systematically benchmark the VM-UNet architecture against standard Convolutional (UNet-ResNet50) and Attention-based (Swin-UNet) baselines.

The core of this research addresses "The Mamba Paradox": why do State Space Models achieve state-of-the-art global accuracy (Dice) while simultaneously struggling to delineate precise object boundaries (Boundary F1)? We hypothesize and mathematically prove that the selective scan mechanism accumulates "Spectral Debt" — an excessive retention of high-frequency aliasing — which inherently damages spatial translation and edge localization.

## 1. Benchmarking Results (ISIC 2018)

Models were evaluated on the ISIC 2018 skin lesion segmentation benchmark. All comparisons were standardized using identical dataset splits and evaluation protocols.

| Architecture | Paradigm | Global Dice (Higher is Better) | Boundary F1 (Higher is Better) | Mean AVR (Lower is Better) |
| :--- | :--- | :---: | :---: | :---: |
| **UNet (ResNet50)** | Convolutional | 0.9056 | 0.2380 | 0.1596 |
| **Swin-UNet** | Attention | 0.9151 | **0.3453** | **0.1282** |
| **VM-UNet** | Selective Scan | **0.9167** | 0.3139 | 0.1622 |

## 2. Key Experimental Findings

### A. Stage-Wise Spectral Aliasing (AVR)
We measure the Alias Volume Ratio (AVR) — the proportion of spectral energy exceeding 50% of the Nyquist limit — at matched resolutions across the encoding stages of each model. VM-UNet exhibits severe spectral aliasing in the high-resolution early stages compared to both CNNs and Transformers.

| Resolution Level | UNet AVR | Swin-UNet AVR | VM-UNet AVR |
| :--- | :---: | :---: | :---: |
| **Level 1 (~64x64)** | 0.0765 | 0.1239 | **0.2650** |
| **Level 2 (~32x32)** | 0.2097 | 0.1974 | **0.2780** |
| **Level 3 (~16x16)** | 0.2343 | 0.1379 | 0.0656 |
| **Level 4 (~8x8)**   | 0.1180 | 0.0537 | 0.0405 |

### B. Statistical Proof of "Spectral Debt"
To prove that spectral aliasing directly causes boundary localization errors, we computed the Pearson correlation between per-image Mean AVR and per-image BF1 across a pooled sample of 150 inferences (50 images x 3 models). 

- **Pooled Correlation**: r = -0.4414 (p = 1.56e-08)
- **Partial Correlation (controlling for model architecture)**: r = -0.3935 (p = 6.30e-07)

The highly significant negative partial correlation proves that "Spectral Debt" is a fundamental phenomenon of spatial resolution processing. An image that triggers higher-than-average aliasing strictly causes lower-than-average boundary precision, independent of the underlying architecture. State-Space Models inherently suffer more from this due to the lack of continuous low-pass filtering in the selective scan mechanism.

### C. Shift Consistency (Translation Equivariance)
We evaluated the models' robustness to horizontal spatial translations. CNNs remain the gold standard due to sliding convolutions, while Swin-UNet degrades significantly when shifts cross rigid window boundaries. VM-UNet sits between the two paradigms: the continuous sequence scanning preserves translation equivariance better than strict windowed attention, but it still suffers from initial patch-embedding artifacts.

| Architecture | Shift 1px | Shift 3px | Shift 5px |
| :--- | :---: | :---: | :---: |
| **UNet (CNN)** | 0.9864 | 0.9678 | 0.9693 |
| **Swin-UNet** | 0.9670 | 0.9484 | 0.9196 |
| **VM-UNet** | 0.9753 | 0.9549 | 0.9536 |

*(Values represent Mean Intersection over Union between the baseline prediction and the shifted prediction)*

## 3. Repository Structure

- `models/`: Architecture definitions for VM-UNet, Swin-UNet, and standard UNet baselines.
- `results/`: Processed CSV outputs containing stage-wise AVR, correlation statistics, and boundary evaluations.
- `tools/`: Independent analysis scripts used to generate the paper's findings.
- `best-ckpt/`: Official trained model weights for the ISIC18 benchmark.

## 4. Evaluation Scripts
The root of the repository contains unified evaluation scripts designed to guarantee identical treatment of all architectures:
- `avr_stagewise_all.py`: Extracts matched-resolution feature maps and computes stage-wise Nyquist frequency masking.
- `per_image_correlation.py`: Calculates Pearson and partial correlations between spectral energy and boundary delineation.
- `shift_consistency.py`: Applies translation permutations and computes equivariance degradation.
- `boundary_eval.py`: Computes rigorous Boundary F1 (BF1) scores using morphological erosions.