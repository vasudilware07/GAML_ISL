# Geometry-Aware Metric Learning for Cross-Lingual Few-Shot Sign Language Recognition

[![GitHub](https://img.shields.io/badge/GitHub-fjkrch%2Fsign__metric__learning-blue?logo=github)](https://github.com/fjkrch/sign_metric_learning)

> Modular, reproducible research framework for **static-image** sign-language
> recognition using metric learning with cross-lingual transfer across four
> sign-language alphabets.
>
> All experiments are deterministic (seed = 42) and fully reproducible from
> a single shell script.  **All reported results use JSON stratified splits
> evaluated on the test split only.**

---

## 1. TL;DR

A geometry-aware **joint-angle representation** (20-D, invariant to rotation,
translation, and scale) combined with Prototypical Networks achieves up to
**95.4%** 5-way 5-shot accuracy on ASL and enables cross-lingual transfer
from ASL to LIBRAS (95.0% frozen, 96.0% target-sup.), Arabic (91.3% frozen,
92.9% target-sup.), and Thai (53.8% frozen, 58.5% target-sup.).

**Key results at a glance (5-way 5-shot, test split, 600 episodes):**

| Dataset | Best Accuracy | Config |
|---------|---------------|--------|
| ASL     | 95.4%         | Transformer / raw_angle |
| LIBRAS  | 94.1%         | MLP / angle |
| Arabic  | 89.8%         | MLP / angle |
| Thai    | 52.7%         | MLP / angle |

---

## 2. Reproducibility Notes

| Item | Value |
|------|-------|
| Python | 3.13.11 |
| PyTorch | 2.10.0+cpu |
| Random seed | 42 (all scripts) |
| Hardware | CPU only (no GPU required) |
| Split ratio | 70 / 30 stratified per class |
| Episodes | 600 per evaluation |
| Deterministic seeding | `rng = np.random.RandomState(seed + e)` per episode |

The full reproduction manifest is documented in `configs/reproduce.yaml`.
Running the commands below (Sections 4–7) with the same data, seed, and code
version will reproduce byte-identical results.

---

## 3. Install & Pin Dependencies

```bash
# Clone
git clone https://github.com/fjkrch/sign_metric_learning.git
cd sign_metric_learning

# Create virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

**Exact tested versions** (see comments in `requirements.txt`):
```
torch==2.10.0  numpy==2.4.1  scipy==1.17.0  scikit-learn==1.8.0
matplotlib==3.10.8  mediapipe==0.10.32  opencv-contrib-python==4.13.0.92
PyYAML==6.0.3  tqdm==4.67.1
```

---

## 4. Data Acquisition

### 4.1 Download from Kaggle

Requires `KAGGLE_API_TOKEN` or `~/.kaggle/kaggle.json`.

```bash
for ds in asl-alphabet thai-fingerspelling libras-alphabet arabic-sign-alphabet; do
    python tools/auto_find_download_and_filter_onehand.py \
        --dataset "$ds" --download --extract --seed 42
done
```

### 4.2 Datasets

| Dataset | Kaggle Slug | Classes | Total Samples |
|---------|-------------|---------|---------------|
| ASL Alphabet | `grassknoted/asl-alphabet` | 29 | ~63 500 |
| LIBRAS Alphabet | `williansoliveira/libras` | 21 | ~34 300 |
| Arabic Sign Alphabet | `muhammadalbrham/rgb-arabic-alphabets-sign-language-dataset` | 31 | ~7 100 |
| Thai Fingerspelling | `nickihartmann/thai-letter-sign-language` | 42 | ~2 900 |

---

## 5. Preprocessing

Extract MediaPipe hand landmarks (21 × 3 `.npy` per image) with
wrist-centring and scale-normalisation:

```bash
python data/preprocess.py --image_dir data/raw/asl_alphabet       --output_dir data/processed/asl_alphabet
python data/preprocess.py --image_dir data/raw/libras_alphabet    --output_dir data/processed/libras_alphabet
python data/preprocess.py --image_dir data/raw/arabic_sign_alphabet --output_dir data/processed/arabic_sign_alphabet
python data/preprocess.py --image_dir data/raw/thai_fingerspelling --output_dir data/processed/thai_fingerspelling
```

After preprocessing, each dataset lives in `data/processed/<name>/<class>/*.npy`.

### Representations

| Repr | Dim | Description |
|------|-----|-------------|
| `raw` | 63 | Flattened (21 × 3) normalised xyz |
| `angle` | 20 | Inter-joint angles from 20 anatomical triplets |
| `raw_angle` | 83 | z-normalised raw ‖ angle |

### Model Architecture

```
┌───────────┐    ┌───────────────┐    ┌──────────────┐    ┌─────────────────────┐
│   Image   │───▶│  MediaPipe    │───▶│ 21 Landmarks │───▶│  Representation     │
│           │    │  Hands v0.10  │    │  (x,y,z)×21  │    │  raw (63-D)         │
└───────────┘    └───────────────┘    └──────────────┘    │  angle (20-D)       │
                                                          │  raw_angle (83-D)   │
                                                          └─────────┬───────────┘
                                                                    │
                                                                    ▼
                                                          ┌─────────────────────┐
                                                          │  Encoder            │
                                                          │  MLP: 256×2 → 128  │
                                                          │  Trans: 2L 4H →128 │
                                                          └─────────┬───────────┘
                                                                    │
                                          ┌─────────────────────────┼──────────────────────┐
                                          │                         │                      │
                                          ▼                         ▼                      ▼
                                   ┌─────────────┐          ┌─────────────┐         ┌────────────┐
                                   │   Frozen    │          │ Target-sup. │         │  ProtoNet  │
                                   │  (encoder   │          │ (last-layer │         │  Head      │
                                   │   fixed)    │          │  fine-tuned │         │            │
                                   └──────┬──────┘          │  on target  │         │ Prototypes │
                                          │                 │  train split│         │  = mean    │
                                          │                 │  NOT episode│         │  embeddings│
                                          │                 │  support)   │         │            │
                                          │                 └──────┬──────┘         │ Query →    │
                                          │                        │               │ nearest    │
                                          └────────────────────────┘               │ prototype  │
                                                          │                        └─────┬──────┘
                                                          ▼                              │
                                                   128-D Embedding ──────────────────────┘
                                                                           │
                                                                           ▼
                                                                   Classification
                                                              (−‖z_q − c_n‖² → softmax)
```

**Encoder parameter counts:**

| Encoder | Repr | Input Dim | Parameters |
|---------|------|-----------|------------|
| MLP | `raw` | 63 | 116,096 |
| MLP | `angle` | 20 | 105,088 |
| MLP | `raw_angle` | 83 | 121,216 |
| Transformer | `raw` | 63 | 282,240 |
| Transformer | `angle` | 20 | 284,416 |
| Transformer | `raw_angle` | 83 | 292,480 |

---

## 6. Split Generation

Splits are **JSON-based, stratified, and deterministic**.  Each class is
independently shuffled with `numpy.RandomState(seed)` and split at
`floor(ratio × n_c)` train samples, with at least 1 sample in each split.

```bash
python tools/make_splits.py --dataset asl_alphabet         --seed 42 --ratio 0.7
python tools/make_splits.py --dataset libras_alphabet      --seed 42 --ratio 0.7
python tools/make_splits.py --dataset arabic_sign_alphabet --seed 42 --ratio 0.7
python tools/make_splits.py --dataset thai_fingerspelling  --seed 42 --ratio 0.7
```

**Output:** `splits/<dataset>_train.json` and `splits/<dataset>_test.json`.

### Split Statistics

| Dataset | Train | Test | Classes | Min test/class |
|---------|-------|------|---------|----------------|
| ASL Alphabet | 44 498 | 19 093 | 29 | 1 |
| LIBRAS Alphabet | 23 993 | 10 291 | 21 | 447 |
| Arabic Sign Alphabet | 4 952 | 2 141 | 31 | 56 |
| Thai Fingerspelling | 1 998 | 863 | 42 | 12 |

### Integrity Guarantees

- No duplicate paths within a split
- Zero overlap between train and test (`validate_no_leak()`)
- At least 1 sample per class in each split

---

## 7. Reproduce All Results

### 7.1 Within-domain evaluation (no pretraining)

```bash
python tools/run_full_matrix.py \
    --datasets asl_alphabet libras_alphabet arabic_sign_alphabet thai_fingerspelling \
    --encoders mlp transformer \
    --representations raw angle raw_angle \
    --shots 1 3 5 --episodes 600 --seed 42 \
    --eval_split test --json_splits --auto_adjust_q \
    --output results/matrix_final.csv
```

### 7.1b Trained within-domain evaluation

This variant first trains the encoder on each dataset's own JSON train split,
then evaluates on the same dataset's JSON test split. It is useful when the
goal is a trained within-domain model rather than the random-init no-pretraining
protocol in Section 9.1.

```bash
python tools/run_within_domain_trained.py \
    --datasets asl_alphabet arabic_sign_alphabet libras_alphabet thai_fingerspelling \
    --encoders mlp transformer \
    --representations raw angle raw_angle \
    --shots 1 3 5 \
    --episodes_train 600 --episodes_eval 600 --epochs 5 \
    --q_query 5 --seeds 42 1337 2024 --device cuda --auto_adjust_q \
    --output results/within_domain_trained_full_gpu_3seeds_600ep.csv --resume
```

### 7.2 Cross-domain evaluation (pretrain on ASL)

```bash
# Full expanded cross-domain eval (all encoder × repr × mode)
python tools/run_cross_domain_expanded.py \
    --encoders mlp transformer \
    --reprs raw angle raw_angle \
    --modes frozen adapted \\
    --episodes 600 --seed 42
```

Or manually pretrain + evaluate single combos:

```bash
# Pretrain on ASL train split
python train.py --config configs/base.yaml --dataset asl_alphabet \
    --json_splits --save results/checkpoints/best_asl_alphabet.pt --epochs 10

# Evaluate cross-domain on each target's test split
for target in asl_alphabet libras_alphabet arabic_sign_alphabet thai_fingerspelling; do
    python evaluate.py --config configs/base.yaml \
        --cross_domain_eval \
        --source_dataset asl_alphabet \
        --target_dataset "$target" \
        --source_ckpt results/checkpoints/best_asl_alphabet.pt \
        --episodes 600 --seed 42 \
        --json_splits --split test --auto_adjust_q
done
```

### 7.3 Multi-source cross-lingual transfer (all 4 languages)

Same entrypoint as Section 7.2, but with `--source` pointing to each
non-default source language. Each run evaluates Source→Target for all 3
targets (excluding same-language self-transfer).

```bash
# Record git commit for reproducibility
git rev-parse HEAD   # 73e0ba7a0c87737c8b5dae48d2adbdbd158791bf

# LIBRAS as source
for target in arabic_sign_alphabet asl_alphabet thai_fingerspelling; do
  python tools/run_cross_domain_expanded.py \
    --source libras_alphabet --targets $target \
    --encoders mlp --reprs raw angle raw_angle \
    --modes frozen adapted --episodes 600 --seed 42 \
    --output results/cross_domain_libras_to_${target%%_*}.csv
done

# Arabic as source
for target in libras_alphabet asl_alphabet thai_fingerspelling; do
  python tools/run_cross_domain_expanded.py \
    --source arabic_sign_alphabet --targets $target \
    --encoders mlp --reprs raw angle raw_angle \
    --modes frozen adapted --episodes 600 --seed 42 \
    --output results/cross_domain_arabic_to_${target%%_*}.csv
done

# Thai as source
for target in asl_alphabet libras_alphabet arabic_sign_alphabet; do
  python tools/run_cross_domain_expanded.py \
    --source thai_fingerspelling --targets $target \
    --encoders mlp --reprs raw angle raw_angle \
    --modes frozen adapted --episodes 600 --seed 42 \
    --output results/cross_domain_thai_to_${target%%_*}.csv
done
```

**Key flags / hyperparameters** (identical to Section 7.2):

| Parameter | Value |
|-----------|-------|
| Pretraining epochs | 3 |
| Pretraining loss | SupCon (τ = 0.07) + NLL, weight 0.5 |
| Pretraining episodes | 100 per epoch |
| Adaptation epochs | 5 (last-layer fine-tune only) |
| Evaluation | 5-way 5-shot, Q = 15, 600 episodes |
| Seed | 42 |
| Optimizer | AdamW (lr = 1e-4, wd = 1e-4) |
| Grad clip | 1.0 |

**Outputs:**
- `results/cross_domain_libras_to_arabic.csv` (6 rows)
- `results/cross_domain_libras_to_asl.csv` (6 rows)
- `results/cross_domain_libras_to_thai.csv` (6 rows)
- `results/cross_domain_arabic_to_libras.csv` (6 rows)
- `results/cross_domain_arabic_to_asl.csv` (6 rows)
- `results/cross_domain_arabic_to_thai.csv` (6 rows)
- `results/cross_domain_thai_to_asl.csv` (6 rows)
- `results/cross_domain_thai_to_libras.csv` (6 rows)
- `results/cross_domain_thai_to_arabic.csv` (6 rows)

### 7.4 Baselines

```bash
# Linear classifier baseline (full train → test)
python tools/run_baselines.py --experiment linear_classifier

# Robustness check with 3 seeds
python tools/run_baselines.py --experiment robustness --seeds 42 1337 2024

# Episode-wise linear head (per-episode logistic regression)
python tools/run_baselines.py --experiment episode_linear

# Input-space nearest prototype (no encoder)
python tools/run_baselines.py --experiment input_space

# All baselines at once
python tools/run_baselines.py --experiment all
```

### 7.5 Normalisation ablation (requires re-preprocessing)

```bash
# Re-preprocess without normalisation
python -c "from data.preprocess import preprocess_dataset; \
    preprocess_dataset('data/raw/arabic-sign-alphabet', 'data/processed/arabic_sign_alphabet_nonorm', \
    normalize_translation=False, normalize_scale=False)"
python -c "from data.preprocess import preprocess_dataset; \
    preprocess_dataset('data/raw/libras-alphabet/train', 'data/processed/libras_alphabet_nonorm', \
    normalize_translation=False, normalize_scale=False)"

# Generate splits for nonorm datasets
python tools/make_splits.py --dataset arabic_sign_alphabet_nonorm --seed 42 --ratio 0.7
python tools/make_splits.py --dataset libras_alphabet_nonorm     --seed 42 --ratio 0.7

# Evaluate
python tools/run_full_matrix.py \
    --datasets arabic_sign_alphabet_nonorm libras_alphabet_nonorm \
    --encoders mlp --representations raw angle --shots 5 \
    --episodes 600 --seed 42 --eval_split test --json_splits --auto_adjust_q
```

### 7.6 Export LaTeX tables from CSVs

```bash
python tools/export_tables.py --outdir paper/tables
# Generates: tab_within.tex, tab_cross.tex, tab_ablation.tex,
#            tab_linear.tex, tab_robust.tex
```

---

## 8. Baselines & Flags Reference

### Encoder × Representation compatibility

| Encoder | `raw` | `angle` | `raw_angle` | `graph` | Status |
|---------|-------|---------|-------------|--------|--------|
| MLP | ✓ | ✓ | ✓ | ✓ | **Evaluated** |
| Transformer | ✓ | ✓ | ✓ | ✓ | **Evaluated** |
| GCN | ✓ | ✗ | ✗ | ✓ | Available (not in reported results) |

### Distance metrics

Pass `--metric euclidean` (default) or `--metric cosine` to
`tools/run_full_matrix.py`.  The ProtoNet forward method supports both.

### Key CLI flags

| Flag | Scripts | Description |
|------|---------|-------------|
| `--json_splits` | train, evaluate, run_full_matrix | Use JSON stratified splits |
| `--auto_adjust_q` | evaluate, run_full_matrix | Auto-lower Q when classes are small |
| `--metric cosine` | run_full_matrix | Cosine distance ProtoNet |
| `--eval_split test` | evaluate, run_full_matrix | Evaluate on test split (default) |
| `--seed 42` | all | Deterministic seed |

### Episodic protocol

- **N-way:** 5 (configurable via `--n_way`)
- **K-shot:** 1, 3, 5 (configurable via `--shots`)
- **Q-query:** 15 (configurable via `--q_query`)
- **Episodes:** 600 (configurable via `--episodes`)
- **Feasibility:** Classes need ≥ K+Q test samples; Thai has 27/42 eligible at K=5 Q=15.

---

## 9. Results

### 9.1 Within-domain (no pretraining)

5-way K-shot, Prototypical Networks, 600 episodes, seed 42, test split:

| Dataset | Encoder | Repr | 1-shot | 3-shot | 5-shot |
|---------|---------|------|--------|--------|--------|
| ASL | MLP | raw | 88.7 ± 0.7 | 93.8 ± 0.5 | 94.9 ± 0.4 |
| ASL | MLP | angle | 81.5 ± 0.9 | 86.8 ± 0.7 | 88.4 ± 0.6 |
| ASL | MLP | raw_angle | 90.4 ± 0.7 | 94.5 ± 0.5 | 95.4 ± 0.4 |
| ASL | Transformer | raw | 74.2 ± 1.0 | 80.3 ± 0.9 | 82.1 ± 0.9 |
| ASL | Transformer | angle | 80.8 ± 0.9 | 86.2 ± 0.7 | 87.7 ± 0.7 |
| ASL | Transformer | raw_angle | **90.8 ± 0.7** | **94.7 ± 0.4** | **95.4 ± 0.4** |
| LIBRAS | MLP | raw | 69.7 ± 1.0 | 78.9 ± 0.8 | 81.2 ± 0.8 |
| LIBRAS | MLP | angle | **89.2 ± 0.7** | **92.9 ± 0.5** | **94.1 ± 0.5** |
| LIBRAS | MLP | raw_angle | 71.6 ± 1.0 | 82.2 ± 0.8 | 84.6 ± 0.8 |
| LIBRAS | Transformer | raw | 61.1 ± 0.9 | 65.9 ± 0.9 | 67.9 ± 0.9 |
| LIBRAS | Transformer | angle | 86.9 ± 0.8 | 91.1 ± 0.6 | 92.5 ± 0.5 |
| LIBRAS | Transformer | raw_angle | 72.8 ± 1.0 | 82.3 ± 0.8 | 84.7 ± 0.7 |
| Arabic | MLP | raw | 51.3 ± 0.9 | 60.6 ± 0.8 | 64.5 ± 0.8 |
| Arabic | MLP | angle | **81.1 ± 0.9** | **88.2 ± 0.6** | **89.8 ± 0.6** |
| Arabic | MLP | raw_angle | 55.5 ± 1.0 | 66.6 ± 0.9 | 71.0 ± 0.9 |
| Arabic | Transformer | raw | 43.1 ± 0.9 | 47.1 ± 1.0 | 49.0 ± 0.9 |
| Arabic | Transformer | angle | 78.3 ± 0.9 | 85.7 ± 0.7 | 87.8 ± 0.6 |
| Arabic | Transformer | raw_angle | 55.6 ± 1.0 | 63.2 ± 0.9 | 67.3 ± 0.9 |
| Thai | MLP | raw | 40.3 ± 0.7 | 47.0 ± 0.7 | 48.8 ± 0.8 |
| Thai | MLP | angle | **46.2 ± 0.8** | **51.2 ± 0.8** | **52.7 ± 0.8** |
| Thai | MLP | raw_angle | 43.4 ± 0.8 | 50.2 ± 0.8 | 51.9 ± 0.8 |
| Thai | Transformer | raw | 33.3 ± 0.7 | 35.7 ± 0.6 | 36.0 ± 0.6 |
| Thai | Transformer | angle | 44.6 ± 0.8 | 50.6 ± 0.8 | 51.8 ± 0.8 |
| Thai | Transformer | raw_angle | 42.3 ± 0.7 | 47.1 ± 0.7 | 48.7 ± 0.8 |

Full 72-row CSV: [results/matrix_final.csv](results/matrix_final.csv).

### 9.1b Within-domain after training

Each row trains the encoder on the dataset's own train split, then evaluates
5-way K-shot Prototypical Networks on the same dataset's test split. This run
uses seeds 42/1337/2024, Q=5, 600 train episodes per epoch, 600 evaluation
episodes, 5 epochs, and CUDA. Cells report mean accuracy +/- seed standard
deviation across the three seeds.

| Dataset | Encoder | Repr | 1-shot | 3-shot | 5-shot |
|---------|---------|------|--------|--------|--------|
| ASL | MLP | raw | 97.92 +/- 0.08 | 98.82 +/- 0.13 | **99.04 +/- 0.12** |
| ASL | MLP | angle | 95.03 +/- 0.59 | 97.55 +/- 0.43 | 97.81 +/- 0.36 |
| ASL | MLP | raw_angle | 97.78 +/- 0.26 | 98.69 +/- 0.02 | 98.80 +/- 0.10 |
| ASL | Transformer | raw | 96.70 +/- 0.68 | 98.62 +/- 0.24 | 98.71 +/- 0.17 |
| ASL | Transformer | angle | 94.34 +/- 0.56 | 96.88 +/- 0.31 | 97.44 +/- 0.19 |
| ASL | Transformer | raw_angle | **98.05 +/- 0.12** | 98.69 +/- 0.05 | 98.85 +/- 0.03 |
| Arabic | MLP | raw | 94.98 +/- 0.07 | 97.59 +/- 0.17 | 98.00 +/- 0.22 |
| Arabic | MLP | angle | 93.24 +/- 0.24 | 95.95 +/- 0.17 | 96.31 +/- 0.16 |
| Arabic | MLP | raw_angle | 95.94 +/- 0.25 | 98.11 +/- 0.09 | 98.38 +/- 0.04 |
| Arabic | Transformer | raw | 93.31 +/- 0.14 | 96.89 +/- 0.41 | 97.11 +/- 0.60 |
| Arabic | Transformer | angle | 93.00 +/- 0.49 | 96.07 +/- 0.22 | 96.38 +/- 0.08 |
| Arabic | Transformer | raw_angle | **96.52 +/- 0.26** | **98.36 +/- 0.02** | **98.59 +/- 0.01** |
| LIBRAS | MLP | raw | 98.22 +/- 0.40 | 99.18 +/- 0.16 | 99.32 +/- 0.10 |
| LIBRAS | MLP | angle | 98.57 +/- 0.13 | 99.23 +/- 0.08 | 99.41 +/- 0.10 |
| LIBRAS | MLP | raw_angle | 98.81 +/- 0.37 | 99.58 +/- 0.10 | 99.72 +/- 0.04 |
| LIBRAS | Transformer | raw | 97.47 +/- 0.40 | 98.13 +/- 0.31 | 98.33 +/- 0.16 |
| LIBRAS | Transformer | angle | 98.15 +/- 0.17 | 99.03 +/- 0.04 | 99.24 +/- 0.07 |
| LIBRAS | Transformer | raw_angle | **99.33 +/- 0.23** | **99.78 +/- 0.07** | **99.85 +/- 0.07** |
| Thai | MLP | raw | 75.39 +/- 1.24 | 83.99 +/- 0.96 | 85.62 +/- 0.71 |
| Thai | MLP | angle | 73.11 +/- 0.51 | 81.72 +/- 0.93 | 83.40 +/- 0.35 |
| Thai | MLP | raw_angle | 78.18 +/- 0.97 | 86.14 +/- 0.62 | 87.25 +/- 0.52 |
| Thai | Transformer | raw | 71.33 +/- 0.82 | 80.38 +/- 1.13 | 82.09 +/- 1.17 |
| Thai | Transformer | angle | 69.04 +/- 0.21 | 79.27 +/- 0.62 | 81.83 +/- 0.54 |
| Thai | Transformer | raw_angle | **79.07 +/- 0.63** | **86.65 +/- 0.71** | **88.05 +/- 0.43** |

Full 216-row CSV: [results/within_domain_trained_full_gpu_3seeds_600ep.csv](results/within_domain_trained_full_gpu_3seeds_600ep.csv).
Three-seed summary: [results/within_domain_trained_full_gpu_3seeds_600ep_summary.csv](results/within_domain_trained_full_gpu_3seeds_600ep_summary.csv).

### 9.2 Cross-domain transfer (pretrained on ASL)

Pretrained on ASL (3 epochs, SupCon), evaluated on each target's
**test split** (5-way 5-shot, Q=15, 600 episodes).
``Frozen'' = encoder weights fixed; ``Target-sup.'' = last-layer fine-tuned on the full target train split (not the episode support set).

**Frozen encoder (best per target):**

| Source → Target | Encoder | Repr | Accuracy | 95% CI |
|-----------------|---------|------|----------|--------|
| ASL → ASL | Transformer | raw_angle | **97.1%** | ±0.3 |
| ASL → LIBRAS | MLP | angle | **95.0%** | ±0.4 |
| ASL → Arabic | MLP | angle | **91.3%** | ±0.5 |
| ASL → Thai | Transformer | raw_angle | **53.8%** | ±0.8 |

**Target-sup. (last-layer fine-tuned on target train split, best per target):**

| Source → Target | Encoder | Repr | Accuracy | 95% CI |
|-----------------|---------|------|----------|--------|
| ASL → LIBRAS | MLP | angle | **96.0%** | ±0.4 |
| ASL → Arabic | MLP | raw_angle | **92.9%** | ±0.5 |
| ASL → Thai | MLP | raw | **58.5%** | ±0.9 |

Full 48-row expanded CSV: [results/cross_domain_all.csv](results/cross_domain_all.csv).

### 9.2b Full Cross-Lingual Transfer Matrix (4×4)

We evaluate all 12 source→target directions (4 sources × 3 targets each)
using the identical protocol (MLP encoder, 3 epochs SupCon pretrain,
5-way 5-shot, Q=15, 600 episodes, seed 42, test split).

#### LIBRAS as source

| Direction | Mode | raw | angle | raw_angle |
|-----------|------|-----|-------|-----------|
| LIBRAS → ASL | Frozen | **93.7 ± 0.5** | 88.3 ± 0.7 | 93.4 ± 0.5 |
| LIBRAS → ASL | Target-sup. | **95.0 ± 0.5** | 89.9 ± 0.6 | 94.4 ± 0.5 |
| LIBRAS → Arabic | Frozen | 89.8 ± 0.6 | 91.2 ± 0.6 | **91.7 ± 0.6** |
| LIBRAS → Arabic | Target-sup. | 91.8 ± 0.6 | 92.2 ± 0.5 | **93.8 ± 0.5** |
| LIBRAS → Thai | Frozen | 51.1 ± 0.8 | 49.5 ± 0.8 | **52.2 ± 0.8** |
| LIBRAS → Thai | Target-sup. | 55.9 ± 0.8 | 55.3 ± 0.8 | **57.4 ± 0.8** |

#### Arabic as source

| Direction | Mode | raw | angle | raw_angle |
|-----------|------|-----|-------|-----------|
| Arabic → ASL | Frozen | **94.1 ± 0.4** | 87.5 ± 0.7 | 93.6 ± 0.5 |
| Arabic → ASL | Target-sup. | **95.5 ± 0.4** | 89.7 ± 0.6 | 95.0 ± 0.4 |
| Arabic → LIBRAS | Frozen | 95.9 ± 0.5 | 94.9 ± 0.5 | **97.1 ± 0.4** |
| Arabic → LIBRAS | Target-sup. | 96.5 ± 0.4 | 95.9 ± 0.4 | **97.4 ± 0.4** |
| Arabic → Thai | Frozen | **54.6 ± 0.9** | 50.0 ± 0.8 | 52.8 ± 0.8 |
| Arabic → Thai | Target-sup. | 58.8 ± 0.9 | 56.6 ± 0.8 | **59.2 ± 0.9** |

#### Thai as source

| Direction | Mode | raw | angle | raw_angle |
|-----------|------|-----|-------|-----------|
| Thai → ASL | Frozen | 94.5 ± 0.4 | 88.7 ± 0.6 | **95.1 ± 0.4** |
| Thai → ASL | Target-sup. | **96.3 ± 0.3** | 89.9 ± 0.6 | 95.9 ± 0.4 |
| Thai → LIBRAS | Frozen | 85.4 ± 0.7 | **94.6 ± 0.5** | 88.4 ± 0.7 |
| Thai → LIBRAS | Target-sup. | 94.9 ± 0.5 | 95.3 ± 0.5 | **96.2 ± 0.4** |
| Thai → Arabic | Frozen | 73.8 ± 0.8 | **90.2 ± 0.5** | 79.4 ± 0.8 |
| Thai → Arabic | Target-sup. | 89.8 ± 0.6 | 92.4 ± 0.5 | **93.9 ± 0.5** |

#### Best frozen accuracy — source comparison

| Target | ASL (source) | LIBRAS (source) | Arabic (source) | Thai (source) |
|--------|-------------|-----------------|-----------------|---------------|
| ASL    | —           | 93.7 ± 0.5      | 94.1 ± 0.4      | **95.1 ± 0.4** |
| LIBRAS | 95.0 ± 0.4  | —               | **97.1 ± 0.4** | 94.6 ± 0.5    |
| Arabic | 91.3 ± 0.5  | **91.7 ± 0.6** | —               | 90.2 ± 0.5    |
| Thai   | 53.2 ± 0.8  | 52.2 ± 0.8      | **54.6 ± 0.9** | —              |

**Interpretation.**
All four source languages yield viable cross-lingual transfer.
Thai→ASL frozen (`raw_angle` 95.1%) is the best overall source for ASL,
surpassing Arabic→ASL (94.1%) by +1.0 pp despite Thai having the fewest
training samples (2,023). Arabic→LIBRAS frozen (`raw_angle` 97.1%) remains
the best for LIBRAS (+2.1 pp over ASL). LIBRAS→Arabic (91.7%) and
Thai→Arabic (90.2%) are competitive with ASL→Arabic (91.3%).
Arabic→Thai (54.6%) edges out ASL→Thai (53.2%) by +1.4 pp.

Transfer is *asymmetric*: Thai→ASL (95.1%) vs ASL→Thai (53.2%) shows
a 42 pp gap — the most extreme in our matrix. Thai is a surprisingly
effective *source* despite being the hardest *target*, suggesting that
its 42-class diversity provides rich pre-training signal even with
limited samples. The `raw` representation dominates transfers *to* ASL
(from LIBRAS/Arabic), while `raw_angle` is best for Thai→ASL and most
other directions.

Raw CSVs:
[results/cross_domain_libras_to_arabic.csv](results/cross_domain_libras_to_arabic.csv),
[results/cross_domain_libras_to_asl.csv](results/cross_domain_libras_to_asl.csv),
[results/cross_domain_libras_to_thai.csv](results/cross_domain_libras_to_thai.csv),
[results/cross_domain_arabic_to_libras.csv](results/cross_domain_arabic_to_libras.csv),
[results/cross_domain_arabic_to_asl.csv](results/cross_domain_arabic_to_asl.csv),
[results/cross_domain_arabic_to_thai.csv](results/cross_domain_arabic_to_thai.csv),
[results/cross_domain_thai_to_asl.csv](results/cross_domain_thai_to_asl.csv),
[results/cross_domain_thai_to_libras.csv](results/cross_domain_thai_to_libras.csv),
[results/cross_domain_thai_to_arabic.csv](results/cross_domain_thai_to_arabic.csv).

### 9.3 Normalisation ablation

5-way 5-shot, MLP encoder, 600 episodes, seed 42:

| Dataset | Repr | No Norm | Normalised | Δ |
|---------|------|---------|------------|---|
| LIBRAS | raw | 76.4 | 81.2 | +4.8 |
| LIBRAS | angle | **94.4** | **94.1** | −0.3 |
| Arabic | raw | 59.1 | 64.5 | +5.4 |
| Arabic | angle | **90.1** | **89.8** | +0.3 |

### 9.4 Linear classifier baseline

MLP encoder → StandardScaler → LogisticRegression(C=1.0) on full train split:

| Dataset | Raw (test) | Angle (test) |
|---------|-----------|-------------|
| ASL | 98.6% | 93.8% |
| LIBRAS | **100.0%** | 99.7% |
| Arabic | 92.7% | 90.9% |
| Thai | 54.9% | 51.7% |

### 9.5 Multi-seed robustness

MLP / angle / 5-shot, seeds 42, 1337, 2024:

| Dataset | Seed 42 | Seed 1337 | Seed 2024 | Mean ± Std |
|---------|---------|-----------|-----------|------------|
| ASL | 88.2 | 89.2 | 89.2 | **88.9 ± 0.6** |
| LIBRAS | 94.1 | 94.6 | 93.6 | **94.1 ± 0.5** |
| Arabic | 90.1 | 90.6 | 89.0 | **89.9 ± 0.8** |
| Thai | 51.8 | 50.5 | 51.4 | **51.2 ± 0.7** |

All standard deviations < 1 pp — results are stable.

---

## 10. Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: splits/...json` | Run `python tools/make_splits.py --dataset <name> --seed 42 --ratio 0.7` |
| `ValueError: Not enough eligible classes` | Pass `--auto_adjust_q` to lower Q, or check that preprocessing completed |
| `No hand detected` during preprocessing | Expected — MediaPipe skips images without detected hands |
| Thai accuracy low (~52%) | Structural limitation: 42 classes, 69 samples/class avg. See Section 9 analysis |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Need GPU | All experiments run on CPU; set `--device cuda` if GPU available |
| Kaggle download fails | Ensure `~/.kaggle/kaggle.json` exists with valid API key |

### Smoke test

```bash
python tools/smoke_test.py          # full check (imports + forward + datasets + CSVs)
python tools/smoke_test.py --quick  # imports + compile only
```

---

## Project Structure

```
sign_metric_learning/
├── configs/
│   ├── base.yaml                # Base hyperparameters
│   ├── asl_to_target.yaml       # Cross-lingual transfer config
│   └── reproduce.yaml           # Reproduction manifest
├── data/
│   ├── __init__.py
│   ├── preprocess.py            # MediaPipe landmark extraction
│   ├── datasets.py              # LandmarkDataset, SplitLandmarkDataset
│   └── episodes.py              # Deterministic episodic N-way K-shot sampler
├── models/
│   ├── __init__.py              # Encoder/model factory (supports --metric)
│   ├── mlp_encoder.py           # MLP encoder (256×2, BN, ReLU, Dropout 0.3)
│   ├── temporal_transformer.py  # Spatial Transformer (2L, 4H, 256 FFN)
│   ├── gcn_encoder.py           # GCN on hand skeleton graph
│   ├── prototypical.py          # Prototypical Networks — evaluated (euclidean / cosine)
│   └── siamese.py               # Siamese & Matching Networks (available, not evaluated)
├── losses/
│   └── supcon.py                # SupCon (used), Triplet & ArcFace (available, not evaluated)
├── utils/
│   ├── seed.py                  # Deterministic seed utilities
│   ├── logger.py                # Console + file logging
│   └── metrics.py               # Accuracy, CI, confusion matrix, t-SNE
├── tools/
│   ├── make_splits.py           # JSON stratified splits
│   ├── run_full_matrix.py       # Encoder × Repr × Shot evaluation matrix
│   ├── run_baselines.py         # Linear, episode-linear, input-space, robustness
│   ├── export_tables.py         # CSV → LaTeX table exporter
│   ├── smoke_test.py            # Repo health check (6 stages)
│   ├── run_pretrain_and_eval.sh # Pretrain → cross-domain eval pipeline
│   └── auto_find_download_and_filter_onehand.py
├── splits/                      # Generated JSON split files
├── results/
│   ├── matrix_final.csv         # Within-domain (72 rows)
│   ├── cross_domain.csv         # Cross-domain (4 rows)
│   ├── cross_domain_libras_to_arabic.csv  # LIBRAS→Arabic (6 rows)
│   ├── cross_domain_arabic_to_libras.csv  # Arabic→LIBRAS (6 rows)
│   ├── baseline_linear.csv      # Linear classifier (8 rows)
│   ├── robustness_seeds.csv     # Multi-seed (4 rows)
│   └── nonorm_ablation.csv      # Normalisation ablation (8 rows)
├── paper/tables/                # Auto-generated LaTeX table fragments
├── train.py                     # Episodic training
├── evaluate.py                  # Evaluation & cross-domain eval
├── adapt.py                     # Few-shot adaptation
├── ablation.py                  # Ablation study
├── requirements.txt
├── CITATION.bib
├── LICENSE                      # MIT
└── README.md
```

---

## Hyperparameters

| Parameter | Default |
|-----------|---------|
| Embedding dim | 128 |
| Optimiser | AdamW |
| Learning rate | 1 × 10⁻⁴ |
| Temperature (SupCon) | 0.07 |
| N-way | 5 |
| K-shot | 1, 3, 5 |
| Q-query | 15 |
| Episodes (eval) | 600 |
| Seed | 42 |
| Train/test ratio | 70 / 30 |

All configurable via YAML in `configs/`.

---

## Citation

```bibtex
@inproceedings{geometry_sign_metric_2026,
  title     = {Geometry-Aware Metric Learning for Cross-Lingual Few-Shot
               Sign Language Recognition},
  author    = {Chyanin},
  booktitle = {Workshop on Sign Language Recognition, CVPR},
  year      = {2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
