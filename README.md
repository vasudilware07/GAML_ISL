# Geometry-Aware Metric Learning for Cross-Lingual Few-Shot Sign Language Recognition

[![Paper](https://img.shields.io/badge/arXiv-2603.09213-b31b1b)](https://arxiv.org/abs/2603.09213)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Implementation of the paper *"Geometry-Aware Metric Learning for Cross-Lingual Few-Shot Sign Language Recognition on Static Hand Keypoints"* by Chamachot & Lertniphonphan (2026).

## Overview

This framework proposes a **20-dimensional SO(3)-invariant inter-joint angle descriptor** derived from MediaPipe hand keypoints for cross-lingual few-shot sign language recognition. The angle representation is provably invariant to rotation, translation, and isotropic scaling, enabling **frozen cross-lingual transfer** that frequently exceeds within-domain accuracy.

### Key Results
- Angle features improve over normalised-coordinate baselines by up to **25 percentage points** within-domain
- Frozen cross-lingual transfer frequently **exceeds** within-domain accuracy
- Lightweight MLP encoder with only **~105k parameters**

## Installation

```bash
# Clone the repository
git clone https://github.com/fjkrch/sign_metric_learning.git
cd sign_metric_learning

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Extract Keypoints from Datasets

Download datasets from Kaggle and extract keypoints:

```bash
# ASL Alphabet (29 classes, ~64k images)
python data/extract_keypoints.py \
    --input_dir path/to/asl-alphabet/asl_alphabet_train \
    --output_dir data/landmarks/asl \
    --dataset asl --seed 42

# LIBRAS (21 classes, ~34k images)
python data/extract_keypoints.py \
    --input_dir path/to/libras \
    --output_dir data/landmarks/libras \
    --dataset libras --seed 42

# Arabic SL (31 classes, ~7k images)
python data/extract_keypoints.py \
    --input_dir path/to/arabic-sl \
    --output_dir data/landmarks/arabic \
    --dataset arabic --seed 42

# Thai Fingerspelling (42 classes, ~2.8k images)
python data/extract_keypoints.py \
    --input_dir path/to/thai-sign \
    --output_dir data/landmarks/thai \
    --dataset thai --seed 42
```

### 2. Train Within-Domain

```bash
# Train MLP encoder with angle features on ASL
python train.py --config configs/default.yaml \
    --dataset asl --repr angle --encoder mlp

# Train with raw_angle on LIBRAS
python train.py --config configs/default.yaml \
    --dataset libras --repr raw_angle --encoder mlp
```

### 3. Evaluate

```bash
# 5-way 5-shot evaluation (600 episodes)
python evaluate.py --config configs/default.yaml \
    --dataset asl --repr angle --encoder mlp \
    --checkpoint checkpoints/asl_mlp_angle_best.pt
```

### 4. Cross-Lingual Transfer

```bash
# Frozen transfer: ASL → LIBRAS
python adapt.py --source asl --target libras \
    --repr angle --encoder mlp --mode frozen

# Target-supervised: ASL → Arabic
python adapt.py --source asl --target arabic \
    --repr angle --encoder mlp --mode target_supervised

# Run both modes
python adapt.py --source asl --target thai \
    --repr angle --encoder mlp --mode both
```

### 5. Run Full Experimental Matrix

```bash
# Within-domain (Table 3)
python tools/run_full_matrix.py

# Cross-lingual transfer (Tables 4, 6)
python tools/run_cross_domain_expanded.py

# Baselines (Table 9)
python tools/run_baselines.py

# Export LaTeX tables
python tools/export_tables.py --results_dir results
```

## Architecture

```
Hand Image → MediaPipe Hands → 21 Keypoints (63-D)
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                 ▼
              Raw (63-D)      Angle (20-D)    Raw+Angle (83-D)
              wrist-centre    20 inter-joint   concatenation
              scale-norm      angles (SO(3)    of both
              flatten         invariant)
                    │                │                 │
                    └────────────────┼────────────────┘
                                     ▼
                    ┌─── MLP Encoder (~105k params) ───┐
                    │  Linear→BN→ReLU→Dropout(0.3) × 2 │
                    │  Linear → 128-D embedding         │
                    └──────────────────────────────────┘
                                     │
                                     ▼
                    ┌─── Prototypical Network Head ────┐
                    │  Support → Class Prototypes       │
                    │  Query → Nearest Prototype (L2)   │
                    │  Softmax(−distances) → Prediction │
                    └──────────────────────────────────┘
```

## Representations

| Name | Dim | Description | Invariant to |
|------|-----|-------------|-------------|
| `raw` | 63 | Wrist-centred, scale-normalised coordinates | Translation, Scale |
| `angle` | 20 | Inter-joint angles from anatomical triplets | **SO(3) rotation**, Translation, Scale |
| `raw_angle` | 83 | Concatenation of raw + angle | Partial (angle component is fully invariant) |

## SO(3) Invariance (Eq. 7)

The angle θₖ between consecutive bones at joint k is computed as:

```
θₖ = arccos(uₖ · vₖ / (‖uₖ‖ · ‖vₖ‖))
```

Under a similarity transform T(p) = sRp + t:

```
(sRuₖ) · (sRvₖ)     s²uₖᵀRᵀRvₖ     uₖ · vₖ
─────────────────  =  ─────────────  =  ────────────
‖sRuₖ‖ · ‖sRvₖ‖     s²‖uₖ‖·‖vₖ‖     ‖uₖ‖ · ‖vₖ‖
```

Translation cancels in displacement vectors; rotation and scaling cancel in the normalised dot product. Verify empirically:

```bash
python tests/test_invariance.py
```

## Project Structure

```
├── configs/          # YAML configuration files
├── data/             # Data pipeline (extraction, representations, datasets)
├── models/           # Encoder architectures + Prototypical Network
├── losses/           # Supervised Contrastive Loss
├── tools/            # Automation scripts for full experimental matrix
├── tests/            # SO(3) invariance tests
├── train.py          # Episodic training loop
├── evaluate.py       # 600-episode evaluation with CI
├── adapt.py          # Cross-lingual adaptation (frozen / target-supervised)
└── requirements.txt  # Dependencies
```

## Citation

```bibtex
@article{chamachot2026geometry,
  title={Geometry-Aware Metric Learning for Cross-Lingual Few-Shot Sign
         Language Recognition on Static Hand Keypoints},
  author={Chamachot, Chayanin and Lertniphonphan, Kanokphan},
  journal={arXiv preprint arXiv:2603.09213},
  year={2026}
}
```

## License

This project is released under the MIT License.
