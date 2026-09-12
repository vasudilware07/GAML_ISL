#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Pretrain on a source dataset → evaluate cross-domain on every target
# ═══════════════════════════════════════════════════════════════════════════
#
#  Usage:
#      bash tools/run_pretrain_and_eval.sh           # defaults
#      bash tools/run_pretrain_and_eval.sh --source asl_alphabet --seed 42
#
#  Steps:
#    1. Generate JSON splits for ALL datasets            (tools/make_splits.py)
#    2. Pretrain on the source dataset (train split)      (train.py)
#    3. Evaluate on each target dataset (test split)      (evaluate.py --cross_domain_eval)
#
#  The script is idempotent: existing splits are overwritten with the same
#  seed, and checkpoints / CSVs are replaced by the new run.
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
SOURCE="asl_alphabet"
SEED=42
RATIO=0.7
CONFIG="configs/base.yaml"
EPOCHS=100
EPISODES=600
DEVICE="auto"
AUTO_Q=""    # set to "--auto_adjust_q" to enable

# ── All datasets ──────────────────────────────────────────────────────────
DATASETS=(
    asl_alphabet
    libras_alphabet
    arabic_sign_alphabet
    thai_fingerspelling
)

# ── Parse arguments ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)       SOURCE="$2";    shift 2 ;;
        --seed)         SEED="$2";      shift 2 ;;
        --ratio)        RATIO="$2";     shift 2 ;;
        --config)       CONFIG="$2";    shift 2 ;;
        --epochs)       EPOCHS="$2";    shift 2 ;;
        --episodes)     EPISODES="$2";  shift 2 ;;
        --device)       DEVICE="$2";    shift 2 ;;
        --auto_adjust_q) AUTO_Q="--auto_adjust_q"; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Pretrain & Cross-Domain Evaluate                          ║"
echo "║  source=$SOURCE  seed=$SEED  ratio=$RATIO                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Step 1: Generate splits ──────────────────────────────────────────────
echo ""
echo "── Step 1: Generate JSON splits ────────────────────────────────"
for ds in "${DATASETS[@]}"; do
    echo "  → $ds"
    python tools/make_splits.py --dataset "$ds" --seed "$SEED" --ratio "$RATIO"
done

# ── Step 2: Pretrain on source (train split) ─────────────────────────────
CKPT_DIR="results/checkpoints"
CKPT_PATH="${CKPT_DIR}/best_${SOURCE}.pt"
mkdir -p "$CKPT_DIR"

echo ""
echo "── Step 2: Pretrain on ${SOURCE} (train split) ─────────────────"

# Build a temporary config override for the source dataset
# We use the base config but override the dataset root to the flat dir
python train.py \
    --config "$CONFIG" \
    --dataset "$SOURCE" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --json_splits \
    --save "$CKPT_PATH" \
    $AUTO_Q

echo "  ✔ Checkpoint saved: $CKPT_PATH"

# ── Step 3: Evaluate cross-domain on each target (test split) ────────────
echo ""
echo "── Step 3: Cross-domain evaluation ─────────────────────────────"

# Remove previous cross-domain CSV so header is rewritten
rm -f results/cross_domain.csv

for target in "${DATASETS[@]}"; do
    echo "  → ${SOURCE} → ${target}"
    python evaluate.py \
        --config "$CONFIG" \
        --cross_domain_eval \
        --source_dataset "$SOURCE" \
        --target_dataset "$target" \
        --source_ckpt "$CKPT_PATH" \
        --episodes "$EPISODES" \
        --device "$DEVICE" \
        --json_splits \
        --split test \
        --seed "$SEED" \
        $AUTO_Q \
    || echo "  ⚠  FAILED: ${SOURCE} → ${target} (continuing)"
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Done. Results: results/cross_domain.csv"
echo "══════════════════════════════════════════════════════════════"
