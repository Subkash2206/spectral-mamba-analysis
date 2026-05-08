# Spectral Fingerprints of Vision Architectures

## TL;DR: Unmasking Spectral Artifacts
This study performs a rigorous spectral audit of Mamba-based architectures (VM-UNet). We discover that the previously reported "Spectral Debt" (a global correlation between aliasing and boundary precision) was largely an **evaluation artifact** caused by uncentered 2D-FFTs (DC component bias). Once mean-centered, the linear correlation between total alias volume and Boundary F1 collapses to near-zero (+0.05). However, we reveal a unique **Dual-Stage Spectral Behavior** in Mamba: it exhibits high aliasing in early encoding stages (Level 1) but transitions to aggressive low-pass filtering in deep stages (Level 4), unlike the more uniform spectral profiles of CNNs and Transformers.

![Evidence Chain](../results/figures/band_decomposition.png)
*1. Spectral Evolution: Mamba transitions from high-frequency debt at Level 1 to aggressive smoothing at Level 4.*

![Spectral Leakage](../results/figures/power_spectrum_grid.png)
*2. Level 1 Leakage: 2D FFTs (mean-centered) reveal Mamba's high-frequency energy accumulation in early stages.*

![Statistical Impact](../results/figures/avr_bf1_scatter.png)
*3. Statistical Correction: Mean-centering removes activation bias, showing that global aliasing does not predict boundary precision (r ≈ 0).*

## Abstract
This repository contains a comparative spectral analysis of three dominant architectural paradigms in medical image segmentation: Convolutional Neural Networks (UNet-ResNet50), Vision Transformers (Swin-Tiny), and State-Space Models (VM-UNet). Following a methodological correction—enforcing mean-centered feature maps to remove DC-offset bias—we re-evaluate the "Spectral Debt" hypothesis. Our findings indicate that while Mamba architectures exhibit significantly higher aliasing in high-resolution stages (Level 1 AVR: 0.46), this does not translate to a global boundary precision deficit. Instead, Mamba demonstrates a sophisticated spectral transition, aggressively filtering high-frequencies in deep layers (Level 4 AVR: 0.13). This suggests that Mamba's edge-localization challenges are stage-specific rather than a consequence of a global spectral bottleneck.

## 1. Key Performance Metrics

Models were evaluated on the held-out ISIC 2018 validation set ($N=519$) using rigorously verified weight loading and standardized preprocessing protocols.

| Architecture | Paradigm | Global Dice (↑) | Boundary F1 (BF1) (↑) | Mean AVR (↓) |
| :--- | :--- | :---: | :---: | :---: |
| **VM-UNet (Mamba)** | Selective Scan | **0.9018** | 0.2291 | **0.2789** |
| **UNet-ResNet50** | Convolutional | 0.8982 | 0.1722 | 0.2943 |
| **Swin-Tiny** | Attention | 0.8976 | **0.2520** | 0.3275 |

## 2. Core Scientific Findings

### A. Dual-Stage Spectral Dynamics
By mean-centering feature maps before computing the 2D-FFT, we isolate genuine high-frequency content from activation magnitude biases. This reveals that Mamba's "Spectral Debt" is concentrated in early stages.

#### Stage-wise Alias Volume Ratio (AVR, mean-centered)
| Resolution Level | UNet AVR | Swin-UNet AVR | VM-UNet AVR |
| :--- | :---: | :---: | :---: |
| **Level 1 (~64x64)** | 0.3419 | 0.3291 | **0.4628** |
| **Level 2 (~32x32)** | 0.3587 | 0.3671 | **0.3812** |
| **Level 3 (~16x16)** | 0.2960 | 0.2493 | **0.1411** |
| **Level 4 (~8x8)**   | 0.1804 | **0.3646** | **0.1307** |

#### Frequency Band Decomposition, Mamba (mean-centered)
| Stage | Low Band (<0.25 Ny) | Mid Band (0.25-0.75 Ny) | High Band (>0.75 Ny) |
| :--- | :---: | :---: | :---: |
| **Level 1 (Early)** | 32.54% | 45.67% | 21.79% |
| **Level 4 (Deep)** | **64.78%** | 31.51% | **3.71%** |

### B. Statistical Correction: The Correlation Collapse
To test the causal link between aliasing and edge precision, we performed a per-image correlation analysis. Once DC-bias is removed, the pooled correlation collapses. 

#### Correlation Results — Mean AVR vs. BF1 (mean-centered FFT)
| Population | Pearson *r* | *p*-value | Significant? |
| :--- | :---: | :---: | :--- |
| **VM-UNet (Mamba)** | +0.3219 | 0.0226 | Yes |
| **Pooled (Overall)** | **+0.0541** | 0.5108 | **No** |
| **Partial (Controlled)**| **+0.0004** | 0.9965 | **No** |

## 3. Visualization Gallery
All figures are automatically generated and saved in `results/figures/`:
*   `band_decomposition.png`: Shows the stage-wise energy shift.
*   `avr_bf1_scatter.png`: The "Correlation Collapse" scatter plot.
*   `power_spectrum_grid.png`: Heatmaps of the power spectra.
*   `shift_consistency_curves.png`: Robustness decay curves.