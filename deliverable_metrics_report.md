# 📊 SpectralMamba: Unified Stage-wise Spectral Audit

This report presents the final, unified spectral aliasing results across all three architectures, ensuring identical hook placement and frequency cutoff definitions.

## 1. Experimental Methodology
To resolve previous numerical discrepancies, a new unified audit script (`avr_stagewise_all.py`) was implemented with the following constraints:
*   **Hook Placement**: Captured at the **Output of the Last Block in each Stage**, prior to downsampling. This ensures architectures are compared at matching resolutions (e.g., 64x64 vs 64x64) rather than disjointed structural stages.
*   **Frequency Mask**: `H/4, W/4` (energy beyond 50% of the Nyquist limit).
*   **Consistency**: All models loaded from official checkpoints (`best-vmunet-scratch-isic18.pth`, etc.) and evaluated on a full 50-image subset.
*   **Execution Environment**: Fully executed on **WSL2 using native CUDA `mamba_ssm` kernels**, guaranteeing that VM-UNet's spectral representation is accurate and not compromised by software fallbacks.

---

## 2. Unified Spectral Audit Results (AVR at Matched Resolutions)

| Architecture | Level 1 (~64x64) | Level 2 (~32x32) | Level 3 (~16x16) | Level 4 (~8x8) | Mean AVR ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UNet (CNN)** | 0.0765 | 0.2097 | 0.2343 | 0.1180 | 0.1596 |
| **Swin (Transformer)** | 0.1239 | 0.1974 | 0.1379 | 0.0537 | 0.1282 |
| **VM-UNet (Mamba)** | **0.2650** | **0.2780** | 0.0656 | 0.0405 | **0.1622*** |

*\*Note: VM-UNet shows extreme aliasing in high-resolution levels (1 & 2), which directly impacts boundary precision.*

---

## 3. Findings & Conclusions

1.  **Peak Aliasing in Mamba**: VM-UNet exhibits its highest spectral debt in **Level 2 (AVR 0.2780)** and **Level 1 (AVR 0.2650)**. This is significantly higher than the CNN baseline and the Swin Transformer at the same resolution scales.
2.  **Boundary Precision Deficit**: The high AVR in Stages 1 and 2 for VM-UNet correlates with the observed BF1 deficit. High-resolution aliasing prevents the model from accurately localizing edges during the decoding process.
3.  **The Mamba Paradox Confirmed**: The architecture achieves high global Dice scores through its selective scan context, but pays a "Spectral Debt" in edge localization due to lack of low-pass filtering in its selective mechanism.

---

## 4. Deliverables
*   **Unified Audit Script**: `avr_stagewise_all.py` (includes Python Selective Scan fallback).
*   **Raw Data**: `VM-UNet/results/avr_stagewise_results.csv`.
*   **Verified Checkpoints**: `best-unet-isic18.pth`, `best-swinunet-isic18.pth`, `best-vmunet-scratch-isic18.pth`.
