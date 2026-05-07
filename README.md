# Spectral Analysis of State Space Models for Vision

Investigating whether Mamba architectures suppress high-frequency spatial information and its effect on boundary segmentation performance.

## 📊 Final Benchmarking Results (ISIC 2018)

| Architecture | Paradigm | Global Dice ↑ | Boundary F1 (BF1) ↑ | Mean AVR (Aliasing) ↓ |
| :--- | :--- | :---: | :---: | :---: |
| **UNet (ResNet50)** | Convolutional | 0.9056 | 0.2380 | 0.1363 |
| **Swin-UNet** | Attention | 0.9151 | **0.3453** | **0.1291** |
| **VM-UNet (Mamba)** | Selective Scan | **0.9167** | 0.3139 | 0.5113 |

### Key Findings:
- **Mamba Paradox**: VM-UNet achieves superior global accuracy (Dice) but struggles with boundary precision (BF1) due to significantly higher spectral aliasing (4x higher AVR than Transformer).
- **Spectral Debt**: We observe a direct correlation between internal aliasing and boundary localization error in State Space Models.

## 🚀 Repository Structure
- `models/`: Architecture definitions for VM-UNet, Swin-UNet, and UNet.
- `results/`: Final training logs and CSV metrics for the ISIC18 benchmark.
- `tools/`: Master spectral audit probes and fair comparison scripts.
- `best-ckpt/`: (Local) Best model checkpoints for all architectures.