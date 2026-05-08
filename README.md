# Spectral Fingerprints of Vision Architectures

## TL;DR: Unmasking Spectral Artifacts
This study performs a rigorous spectral audit of Mamba-based architectures (VM-UNet). We discover that the previously reported "Spectral Debt" (a global correlation between aliasing and boundary precision) was largely an **evaluation artifact** caused by uncentered 2D-FFTs (DC component bias). Once mean-centered, the linear correlation between total alias volume and Boundary F1 collapses to near-zero (+0.05). However, we reveal a unique **Dual-Stage Spectral Behavior** in Mamba: it exhibits high aliasing in early encoding stages (Level 1) but transitions to aggressive low-pass filtering in deep stages (Level 4), unlike the more uniform spectral profiles of CNNs and Transformers.

![Evidence Chain](figures/band_decomposition.png)
*1. Spectral Evolution: Mamba transitions from high-frequency debt at Level 1 to aggressive smoothing at Level 4.*

![Spectral Leakage](figures/power_spectrum_grid.png)
*2. Level 1 Leakage: 2D FFTs (mean-centered) reveal Mamba's high-frequency energy accumulation in early stages.*

![Statistical Impact](figures/avr_bf1_scatter.png)
*3. Statistical Correction: Mean-centering removes activation bias, showing that global aliasing does not predict boundary precision (r ≈ 0).*

## Abstract
This repository contains a comparative spectral analysis of three dominant architectural paradigms in medical image segmentation: Convolutional Neural Networks (UNet-ResNet50), Vision Transformers (Swin-Tiny), and State-Space Models (VM-UNet). Following a methodological correction—enforcing mean-centered feature maps to remove DC-offset bias—we re-evaluate the "Spectral Debt" hypothesis. Our findings indicate that while Mamba architectures exhibit significantly higher aliasing in high-resolution stages (Level 1 AVR: 0.44), this does not translate to a global boundary precision deficit. Instead, Mamba demonstrates a sophisticated spectral transition, aggressively filtering high-frequencies in deep layers (Level 4 AVR: 0.14). This suggests that Mamba's edge-localization challenges are stage-specific rather than a consequence of a global spectral bottleneck.

## 1. Key Performance Metrics

Models were evaluated on the ISIC 2018 skin lesion segmentation benchmark using corrected weight loading (`strict=True`) and standardized protocols.

| Architecture | Paradigm | Global Dice | Boundary F1 (BF1) | Mean AVR |
| :--- | :--- | :---: | :---: | :---: |
| **UNet-ResNet50** | Convolutional | 0.9055 | 0.2661 | 0.2943 |
| **Swin-Tiny** | Attention | 0.9146 | **0.3736** | 0.3275 |
| **VM-UNet (Mamba)** | Selective Scan | 0.9086 | 0.3149 | **0.2834** |

## 2. Core Scientific Findings

### A. Dual-Stage Spectral Dynamics
By mean-centering feature maps before computing the 2D-FFT, we isolate genuine high-frequency content from activation magnitude biases. This reveal's Mamba's "Spectral Debt" is concentrated in early stages.

> [!IMPORTANT]
> **Methodological Fix**: All results below use mean-centered FFTs. Previous findings claiming a strong negative correlation between AVR and BF1 were biased by the DC component (activation magnitude), not spectral aliasing itself.

#### Stage-wise Alias Volume Ratio (AVR, mean-centered)
| Resolution Level | UNet AVR | Swin-UNet AVR | VM-UNet AVR |
| :--- | :---: | :---: | :---: |
| **Level 1 (~64x64)** | 0.3419 | 0.3291 | **0.4391** |
| **Level 2 (~32x32)** | 0.3587 | 0.3671 | **0.4012** |
| **Level 3 (~16x16)** | 0.2960 | 0.2493 | **0.1563** |
| **Level 4 (~8x8)**   | 0.1804 | **0.3646** | **0.1370** |

#### Frequency Band Decomposition, Mamba (mean-centered)
| Stage | Low Band (<0.25 Ny) | Mid Band (0.25-0.75 Ny) | High Band (>0.75 Ny) |
| :--- | :---: | :---: | :---: |
| **Level 1 (Early)** | 32.54% | 45.67% | 21.79% |
| **Level 4 (Deep)** | **64.78%** | 31.51% | **3.71%** |

Mamba starts with high spectral leakage (21.8% high-band energy) but ends as the most aggressive low-pass filter among all tested architectures (only 3.7% high-band energy at the bottleneck).

### B. Statistical Correction: The Correlation Collapse
To test the causal link between aliasing and edge precision, we performed a per-image correlation analysis (n=150).

#### Correlation Results — Mean AVR vs. BF1 (mean-centered FFT)
| Population | Pearson *r* | *p*-value | Significant? |
| :--- | :---: | :---: | :--- |
| **UNet (CNN)** | -0.2003 | 0.1632 | No |
| **Swin-UNet** | -0.1303 | 0.3671 | No |
| **VM-UNet (Mamba)** | +0.3219 | 0.0226 | Yes |
| **Pooled (n=150)** | **+0.0541** | 0.5108 | **No** |
| **Partial (Controlled)**| **+0.0004** | 0.9965 | **No** |

Once DC-bias is removed, the pooled correlation collapses. The "Spectral Debt" hypothesis as a global predictor of boundary failure is **refuted**. Spectral aliasing is an architectural trait, but not the primary driver of boundary precision across different paradigms.

### C. Internal Mamba Sensitivity
Interestingly, while the global correlation is zero, Mamba internally shows a **positive** correlation (+0.32, p=0.02). This suggests that within the Mamba architecture, images that retain higher spectral energy (less aliasing/smoothing) actually achieve *better* boundary scores, potentially because the model benefits from richer high-frequency cues when it manages to preserve them.

### D. Shift Consistency
Mamba maintains competitive shift consistency despite its high Level 1 AVR, outperforming Swin-UNet at higher shift magnitudes.

#### Translation Equivariance Results (Mean IoU)
| Architecture | 1px Shift | 3px Shift | 5px Shift |
| :--- | :---: | :---: | :---: |
| **UNet (CNN)** | 0.9864 | 0.9678 | **0.9693** |
| **VM-UNet (Mamba)** | 0.9755 | 0.9551 | 0.9532 |
| **Swin-UNet** | 0.9670 | 0.9484 | 0.9196 |

## 3. Visualization Gallery

### Frequency Band Decomposition
![Band Decomposition](figures/band_decomposition.png)

### Power Spectrum Heatmaps
![Power Spectrum Grid](figures/power_spectrum_grid.png)

### Spectral Aliasing vs. Boundary Precision
![AVR vs BF1 Scatter](figures/avr_bf1_scatter.png)

### Translation Equivariance Curves
![Shift Consistency Curves](figures/shift_consistency_curves.png)

### Stage-wise AVR Distribution
![Stagewise AVR Bars](figures/stagewise_avr_bars.png)

## 4. Reproducibility

### Repository Structure
- `VM-UNet/`: Core Mamba implementation and results submodule.
- `Swin-Unet/`: Transformer baseline submodule.
- `results/`: Consolidated CSV datasets for AVR, BF1, and correlations.

### Analysis Pipeline
1. `avr_stagewise_all.py`: Unified stage-wise spectral audit.
2. `per_image_correlation.py`: Statistical analysis of AVR vs BF1.
3. `shift_consistency.py`: Translation equivariance testing.
4. `visualizations.py`: Generation of all figures.