# TGCN-cVAE+MoE for 3D Human-Furniture Interaction

> **XYZ-only training source:** See [TRAINING_XYZ_ONLY.md](TRAINING_XYZ_ONLY.md) for the 2026-09-25 source snapshot, exact input columns, two training objectives, run commands, and checkpoint provenance. The training instructions below describe the earlier repository setup.

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)
![Viser](https://img.shields.io/badge/Viser-3D_Viz-orange.svg)

Official PyTorch implementation for **Adaptive Spatio-Temporal Graph Convolutional Conditional VAE with Mixture of Experts (TGCN-cVAE+MoE)**.
This model predicts the **Interaction Intensity Weight (IIW)** between 3D human body joints and the surrounding furniture environment using simulated ray-casting.

## 🚀 Features
- **Dynamic Bipartite Graph**: The graph adjacency matrix between human joints (15 nodes) and environmental rays (540 nodes) is dynamically computed at every frame using **Inverse Euclidean Distance**.
- **TGCN Encoder**: Captures both spatial correlation (adaptive GCN) and temporal continuity (Dilated TCN) simultaneously.
- **Mixture of Experts (MoE) Decoder**: Resolves the extreme sparsity issue of IIW data by assigning specialized decoders to different interaction patterns.
- **Cyclical KL Annealing**: Solves the posterior collapse problem commonly found in sparse conditional variables.

## 📁 Repository Structure

```text
.
├── baselines/                 # Ablation studies and baseline models
│   ├── MLP_cVAE_MoE/
│   ├── LSTM_cVAE_MoE/
│   ├── GCN_cVAE_MoE/
│   ├── TCN_cVAE_MoE/
│   └── TGCN_cVAE_TGCN/        # Ablation on decoder (TGCN instead of MoE)
│
├── proposed/                  # ⭐ Proposed Model (Ours)
│   └── TGCN_cVAE_MoE/
│       ├── model.py           # Core architecture
│       ├── train.py           # Training loop with TensorBoard
│       └── eval.py            # Extracts MAE, Peak MAE, Jittering, FLOPs
│
├── utils/                     # Shared components
│   ├── dataset.py             # Memory-mapped DataLoader & 8:1:1 split generator
│   └── graph.py               # Bipartite Adaptive Adjacency Matrix
│
├── data/                      # Isolated evaluation dataset (Unseen during training)
├── Extraction/                # Raw dataset sequences for building mmap
├── visualize.py               # 3D interactive GUI (Powered by Viser)
└── run_all_baselines.sh       # Automated bash script to train all baselines sequentially
```

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/YourUsername/YourRepository.git
cd YourRepository
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏋️‍♂️ Training

Before training, ensure you have built the memory-mapped dataset (`dataset_mmap.npy`) using the utility script (not included in the Git repo due to size constraints).

To train the proposed model:
```bash
python proposed/TGCN_cVAE_MoE/train.py \
    --mmap_path ./dataset_mmap.npy \
    --save_dir ./proposed/TGCN_cVAE_MoE/weights \
    --epochs 50 \
    --batch_size 32 \
    --num_workers 8
```
You can monitor the training progress via TensorBoard:
```bash
tensorboard --logdir ./proposed/TGCN_cVAE_MoE/weights/logs
```

## 📊 Evaluation

To evaluate the best model on the held-out test split:
```bash
python proposed/TGCN_cVAE_MoE/eval.py \
    --mmap_path ./dataset_mmap.npy \
    --weights_path ./proposed/TGCN_cVAE_MoE/weights/best.pth \
    --batch_size 16
```
This script outputs the Overall IIW MAE, Active IIW MAE (focused on sparse contact points), Temporal Jittering, and computational efficiency (FLOPs / Inference Speed).

## 🎮 3D Visualization

We provide an interactive 3D web GUI powered by `viser` to intuitively inspect how the human skeleton interacts with the furniture rays.

```bash
python proposed/TGCN_cVAE_MoE/visualize.py \
    --weights_path ./proposed/TGCN_cVAE_MoE/weights/best.pth \
    --data_dir ./data/ \
    --port 8080
```
Open **`http://localhost:8080`** in your web browser. 
- You can play/pause the sequence time loop.
- You can select specific body parts (e.g., Pelvis, Spine, Hands, Feet) to visualize independent IIW heatmap rays.
