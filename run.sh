#!/usr/bin/env bash
# =============================================================================
# Geometry-Aware Metric Learning for Cross-Lingual Few-Shot Sign Language
# Recognition — Full Experiment Pipeline
# =============================================================================
set -euo pipefail

echo "============================================================"
echo " Sign Metric Learning — Full Pipeline"
echo "============================================================"

# ── 0. Setup ─────────────────────────────────────────────────────────────────
mkdir -p results/checkpoints results/plots results/cross_lingual/checkpoints

# ── 1. Preprocessing (skip if already done or using synthetic data) ──────────
echo ""
echo "[Step 1] Preprocessing"
echo "  (Skipping — using synthetic data for reproducibility demo."
echo "   For real data, run:"
echo "     python data/preprocess.py --image_dir data/raw/asl_alphabet --output_dir data/processed/asl"
echo "     python data/preprocess.py --image_dir data/raw/thai_fingerspelling --output_dir data/processed/thai"
echo "  )"

# ── 2. Pretrain on ASL ──────────────────────────────────────────────────────
echo ""
echo "[Step 2] Pretraining on ASL"
python train.py --config configs/base.yaml --epochs 5
echo "  Done."

# ── 3. Zero-shot evaluation on BdSL ─────────────────────────────────────────
echo ""
echo "[Step 3] Zero-shot evaluation on BdSL"
python evaluate.py --dataset Thai --zero-shot --config configs/base.yaml --episodes 100
echo "  Done."

# ── 4. Few-shot adaptation (1-shot) ─────────────────────────────────────────
echo ""
echo "[Step 4] 1-shot adaptation on Thai"
python adapt.py --dataset Thai --shot 1 --mode finetune_last --config configs/base.yaml
echo "  Done."

# ── 5. Few-shot adaptation (5-shot) ───────────────────────────────────────
echo ""
echo "[Step 5] 5-shot adaptation on Thai"
python adapt.py --dataset Thai --shot 5 --mode finetune_last --config configs/base.yaml
echo "  Done."

# ── 6. Ablation study ───────────────────────────────────────────────────────
echo ""
echo "[Step 6] Ablation study"
python ablation.py --config configs/base.yaml --fast
echo "  Done."

# ── 7. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Pipeline complete. Results saved under results/"
echo "============================================================"
echo ""
echo "Output files:"
echo "  results/zero_shot.csv"
echo "  results/few_shot.csv"
echo "  results/ablation.csv"
echo "  results/plots/tsne*.png"
echo "  results/checkpoints/best_asl.pt"
echo ""
