# Geometry-Aware Metric Learning for Cross-Lingual Few-Shot Sign Language Recognition
## A Complete Paper Explanation

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Why Existing Approaches Fail](#2-why-existing-approaches-fail)
3. [The Core Idea](#3-the-core-idea)
4. [Mathematical Foundation](#4-mathematical-foundation)
5. [The Full Pipeline](#5-the-full-pipeline)
6. [Representations Explained](#6-representations-explained)
7. [Model Architecture](#7-model-architecture)
8. [Training Strategy](#8-training-strategy)
9. [Few-Shot Evaluation Protocol](#9-few-shot-evaluation-protocol)
10. [Cross-Lingual Transfer](#10-cross-lingual-transfer)
11. [Key Experimental Results](#11-key-experimental-results)
12. [Why It Works — Intuition](#12-why-it-works--intuition)
13. [Limitations and Future Work](#13-limitations-and-future-work)
14. [Glossary](#14-glossary)

---

## 1. The Problem

### The Data Bottleneck in Sign Language Recognition

There are over **300 sign languages** used worldwide by more than 70 million Deaf
individuals. Building a recognition system for any one of these languages typically
requires **thousands of labelled examples per class** — an expensive, time-consuming
process that most communities cannot afford.

**The key question this paper asks:**

> Can we train a model on a data-rich sign language (like ASL) and then adapt it
> to recognize signs in a completely different language (like Thai or Arabic)
> using only **5 examples per class**?

This is the problem of **cross-lingual few-shot transfer**.

### What Makes This Hard

```
Challenge 1: Few Examples
    Traditional ML needs 1000s of examples → This paper uses only 5

Challenge 2: Different Languages
    ASL, LIBRAS, Arabic SL, and Thai use different handshapes for different letters

Challenge 3: Different Recording Conditions
    Each dataset was captured with different cameras, angles, lighting, and distances
    → This creates "domain shift" that confuses models
```

---

## 2. Why Existing Approaches Fail

### The Domain Shift Problem

When you extract hand keypoints (21 3D joint positions) from images using MediaPipe,
the raw (x, y, z) coordinates are affected by:

| Factor | Effect on Coordinates | Example |
|--------|----------------------|---------|
| **Camera viewpoint** | Rotation of all coordinates | Frontal vs. side view |
| **Hand distance** | Scale of all coordinates | Close-up vs. far away |
| **Hand position** | Translation of all coordinates | Center vs. corner of frame |

Even though two people are making the **exact same hand sign**, their raw keypoint
coordinates will look completely different if recorded under different conditions.

### Why This Kills Few-Shot Learning

In few-shot learning with Prototypical Networks, each class is represented by a
**prototype** — the average (centroid) of just K=5 example embeddings.

```
With extrinsic variance (camera differences):

    Prototype = mean of 5 noisy examples
                       ↓
    Noise from camera setup >> Signal from hand shape
                       ↓
    Prototype is unreliable → Classification fails
```

This is especially catastrophic in cross-lingual transfer, where source and target
datasets were recorded under entirely different conditions.

---

## 3. The Core Idea

### Angles Don't Change When You Move the Camera

The paper's key insight is elegantly simple:

> **The angle between two bones at a joint is the same regardless of how you
> rotate, translate, or scale the hand.**

Think about it physically:
- If you bend your index finger at 90 degrees, that angle is 90 degrees whether
  someone photographs it from the front, the side, or from across the room.
- The (x,y,z) coordinates of your finger joints change drastically with camera
  position, but the **angle at the joint** does not.

This is called **SO(3) invariance** — invariance to the group of 3D rotations,
plus translation and scaling.

### From 63 Numbers to 20 Numbers

```
Raw coordinates:  21 joints x 3 coordinates = 63 numbers
                  ↓ Affected by camera position, rotation, scale
                  ↓ Different across datasets

Joint angles:     20 inter-joint angles = 20 numbers
                  ↓ NOT affected by camera position, rotation, scale
                  ↓ Same across datasets = PORTABLE
```

By converting keypoints to angles, the paper **eliminates domain shift at the
representation level** — before any learning even happens.

---

## 4. Mathematical Foundation

### 4.1. Hand Skeleton Topology

MediaPipe detects 21 hand landmarks. The skeleton forms a tree rooted at the wrist:

```
                            Wrist (0)
                          /   |   |   \    \
                        /     |   |     \    \
                    Thumb   Index Mid   Ring  Pinky
                    1-4     5-8  9-12  13-16 17-20

Each finger is a kinematic chain of 4 joints:
    Wrist → MCP → PIP → DIP → Tip

Example (Index finger):
    0 → 5 → 6 → 7 → 8
    (wrist → MCP → PIP → DIP → tip)
```

### 4.2. Raw Coordinate Preprocessing (Equations 1-3)

**Step 1: Wrist-centering (remove translation)**

    p_hat_i = p_i - p_0      for all i in {0, ..., 20}

Subtract the wrist position from all keypoints so the wrist is at the origin.
This removes the effect of where the hand appears in the image.

**Step 2: Scale normalization (remove scale)**

    p_tilde_i = p_hat_i / max_{j,k} ||p_hat_j - p_hat_k||

Divide all coordinates by the maximum pairwise distance. This removes the effect
of how close the hand is to the camera.

**Step 3: Flatten**

    x_raw = flatten(p_tilde_0, ..., p_tilde_20) ∈ R^63

Stack all coordinates into a single 63-dimensional vector.

> **Important limitation**: Even after these normalizations, raw coordinates are
> still affected by 3D rotation (camera viewpoint). Only angles fix this.

### 4.3. Angle Computation (Equations 4-7)

For each joint that connects two bones, we compute the angle between those bones.

**Step 1: Define anatomical triplets**

Each triplet (a, j, c) consists of:
- a = parent keypoint (ancestor)
- j = pivot keypoint (where the angle is measured)
- c = child keypoint

Example: Triplet (5, 6, 7) measures the angle at the PIP joint of the index
finger, with MCP (5) as parent, PIP (6) as pivot, and DIP (7) as child.

**Step 2: Compute displacement vectors**

    u_k = p_{a_k} - p_{j_k}    (vector from pivot to parent)
    v_k = p_{c_k} - p_{j_k}    (vector from pivot to child)

**Step 3: Compute the angle via normalized dot product**

    theta_k = arccos( (u_k . v_k) / (||u_k|| * ||v_k||) )

This gives the angle in radians between the two bone directions at each joint.

### 4.4. The Invariance Proof (Equation 7)

This is the mathematical heart of the paper. Under a similarity transform
T(p) = sRp + t, where:
- R is a rotation matrix (SO(3))
- s is a positive scalar (isotropic scale)
- t is a translation vector

The angle computation becomes:

```
    (sR*u_k) . (sR*v_k)       s^2 * u_k^T * R^T * R * v_k
    ─────────────────────  =   ────────────────────────────
    ||sR*u_k|| * ||sR*v_k||   s^2 * ||u_k|| * ||v_k||

Since R^T * R = I (rotation matrices are orthogonal):

                               u_k . v_k
                           =   ──────────────
                               ||u_k|| * ||v_k||
```

**Translation cancels** because it disappears when computing differences (u = p_a - p_j).
**Scale cancels** because s^2 appears in both numerator and denominator.
**Rotation cancels** because R^T * R = I (the identity matrix).

Therefore: **theta_k is exactly the same** regardless of rotation, translation, or scale.
This is not an approximation — it is a mathematical proof.

### 4.5. Combined Representation (Equation 6)

    x_raw_angle = [x_raw ; x_angle] ∈ R^83

Concatenate the 63-D raw coordinates with the 20-D angles to get an 83-D vector
that captures both positional and angular information.

---

## 5. The Full Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  Hand Image  │────→│  MediaPipe   │────→│  21 Keypoints  │
│              │     │  Hands v0.10 │     │  (21 x 3)      │
└─────────────┘     └──────────────┘     └───────┬────────┘
                                                  │
                           ┌──────────────────────┼──────────────────────┐
                           │                      │                      │
                           ▼                      ▼                      ▼
                    ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
                    │  Raw (63-D)  │     │ Angle (20-D) │     │ Raw+Angle (83-D) │
                    │  Wrist-center│     │  20 joint    │     │  Concatenation   │
                    │  + Scale-norm│     │  angles      │     │  of both         │
                    └──────┬───────┘     └──────┬───────┘     └────────┬─────────┘
                           │                    │                      │
                           └────────────────────┼──────────────────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │   Encoder (MLP or     │
                                    │   Transformer)        │
                                    │   → 128-D embedding   │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │  Prototypical Network │
                                    │  Nearest prototype    │
                                    │  classification       │
                                    └───────────────────────┘
```

---

## 6. Representations Explained

### Raw (63-D)

```python
# Pseudocode
keypoints -= keypoints[0]                     # Wrist-center
keypoints /= max_pairwise_distance(keypoints) # Scale-normalize
x_raw = keypoints.flatten()                   # → 63-D vector
```

**Pros**: Preserves absolute spatial information (finger positions relative to each other).
**Cons**: Still affected by camera rotation (3D viewpoint changes).

### Angle (20-D)

```python
# Pseudocode — for each anatomical triplet (parent, pivot, child):
u = keypoints[parent] - keypoints[pivot]     # Vector to parent
v = keypoints[child]  - keypoints[pivot]     # Vector to child
angle = arccos(dot(u, v) / (norm(u) * norm(v)))  # Angle at joint
```

**Pros**: Completely invariant to rotation, translation, and scale. Portable across datasets.
**Cons**: Discards absolute position and bone-length information.

### Raw+Angle (83-D)

```python
x_combined = concatenate(x_raw, x_angle)     # Best of both worlds
```

**Pros**: Captures both positional and angular cues.
**Best for**: When domain shift is moderate and data is sufficient (e.g., ASL, LIBRAS↔Arabic).

### When to Use Which?

| Scenario | Best Representation | Why |
|----------|-------------------|-----|
| Large domain shift (e.g., ASL→Arabic) | **angle** | Removes all extrinsic variance |
| Small domain shift + abundant data | **raw_angle** | Complementary cues help |
| Uniform capture conditions (e.g., ASL alone) | **raw_angle** | Positional info is reliable |
| Privacy-sensitive deployment | **angle** | Only stores angles, not positions |

---

## 7. Model Architecture

### MLP Encoder (~105k parameters for angle input)

```
Input (20-D / 63-D / 83-D)
    │
    ▼
┌─────────────────────────────┐
│ Linear(input_dim → 256)     │
│ BatchNorm1d(256)            │
│ ReLU                        │
│ Dropout(0.3)                │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Linear(256 → 256)           │
│ BatchNorm1d(256)            │
│ ReLU                        │
│ Dropout(0.3)                │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Linear(256 → 128)           │  ← Output embedding
└─────────────────────────────┘
```

**Parameter counts (matches Table 1 of the paper exactly):**

| Representation | Input Dim | Parameters |
|---------------|-----------|-----------|
| raw | 63 | 116,096 |
| angle | 20 | 105,088 |
| raw_angle | 83 | 121,216 |

**Why MLP works well here**: The paper shows that simple encoders suffice when the
input representation is well-designed. This is consistent with Chen et al. (2019):
a good feature representation matters more than a complex model.

### Transformer Encoder (~282k parameters)

For the **raw** representation:
```
21 keypoints (each 3-D) → project to 128-D tokens
    + sinusoidal positional encodings
    → 2 Transformer encoder layers (4 heads, FFN=256)
    → mean-pool across 21 tokens
    → project to 128-D
```

For **angle** and **raw_angle**: The full vector is treated as a single token,
so self-attention has nothing to attend across — it degenerates to a feedforward
path, making the Transformer functionally equivalent to a deeper MLP.

**Key finding**: The MLP generally matches or outperforms the Transformer for this
task, confirming that model complexity is less important than representation quality.

### Prototypical Network Head

No additional learnable parameters. Classification works as follows:

```
1. ENCODE:     z = f_phi(x)          Map features to 128-D embedding

2. PROTOTYPE:  c_n = (1/K) * sum(z_i for class n)
                                      Mean of K support embeddings per class

3. DISTANCE:   d(z_q, c_n) = ||z_q - c_n||^2
                                      Squared Euclidean distance

4. CLASSIFY:   y_hat = argmin_n d(z_q, c_n)
                                      Assign to nearest prototype

5. PROBABILITIES:  P(y=n|x_q) = softmax(-d(z_q, c_n))
                                      Soft class probabilities
```

---

## 8. Training Strategy

### Combined Loss Function

The training objective is a weighted combination of two losses:

```
L_total = L_episode + 0.5 * L_supcon
```

**Episode Classification Loss (L_episode)**:
- Standard negative log-likelihood on the ProtoNet's predicted probabilities.
- Teaches the model to place same-class embeddings near each other and different-class
  embeddings far apart within each episode.

**Supervised Contrastive Loss (L_supcon)**:
- Khosla et al. (2020): Pulls together embeddings of the same class while pushing
  apart embeddings of different classes.
- Temperature parameter tau = 0.07 controls the concentration of the distribution.
- Weight lambda = 0.5 balances it with the episode loss.

### Optimization Details

```
Optimizer:        AdamW (lr = 1e-4, weight_decay = 1e-4)
LR Schedule:      Cosine annealing over max_epochs
Gradient Clipping: Max norm = 1.0
Early Stopping:   Patience = 15 epochs (stop if no improvement)
```

### Training is Episodic

Instead of standard batch training, the model trains on **episodes**:

```
For each episode:
    1. Sample 5 random classes (from available classes)
    2. For each class:
       - Draw K=5 support examples
       - Draw Q=15 query examples
    3. Compute prototypes from support set
    4. Classify query set
    5. Compute loss and backpropagate
```

This simulates the few-shot test condition during training.

---

## 9. Few-Shot Evaluation Protocol

### Episode Definition

```
N-way:   5 classes per episode
K-shot:  K support examples per class (K ∈ {1, 3, 5})
Q-query: 15 query examples per class

Total per episode: 5 * (K + 15) samples
```

### Deterministic Reproducibility

Each episode is seeded with `seed + episode_index`:
- Episode 0 uses seed 42
- Episode 1 uses seed 43
- ...
- Episode 599 uses seed 641

This ensures **exact reproducibility** — anyone running the same code gets
identical results.

### Statistical Reporting

```
Mean accuracy over 600 episodes ± 95% confidence interval

CI_95 = 1.96 * std / sqrt(600)

Example: 94.1 ± 0.5% means the true accuracy lies between 93.6% and 94.6%
with 95% confidence.
```

### Eligibility

Not all classes can participate in evaluation. A class needs at least K + Q
samples in the test split:

| K-shot | Min test samples needed | Reason |
|--------|------------------------|--------|
| K=1 | 16 | 1 support + 15 query |
| K=3 | 18 | 3 support + 15 query |
| K=5 | 20 | 5 support + 15 query |

Thai has only 27/42 eligible classes at K=5 (some have too few test samples).

---

## 10. Cross-Lingual Transfer

### The Central Experiment

```
Source language (data-rich)              Target language (low-resource)
┌─────────────────────┐                 ┌─────────────────────┐
│    ASL (29 classes)  │    Transfer     │  Arabic (31 classes) │
│    44,498 train      │ ──────────→    │  Only 5 examples     │
│    samples           │                │  per class needed    │
└─────────────────────┘                 └─────────────────────┘
```

### Two Adaptation Modes

**1. Frozen Transfer**
```
Train encoder on source language → Fix ALL weights → Evaluate on target
```
- Tests: How portable is the learned embedding?
- No target-language training at all
- The purest test of cross-lingual generalization

**2. Target-Supervised Transfer**
```
Train encoder on source → Freeze backbone → Fine-tune LAST layer on target train split
```
- Fine-tune for up to 20 epochs with lr=1e-4
- Only the final 128-D projection layer is updated
- Minimal adaptation with minimal risk of overfitting

### Key Insight: Angle Features Enable Transfer

```
Raw coordinates:   Encode camera-specific information
                   → Source and target have different "camera fingerprints"
                   → Prototypes are contaminated by camera differences
                   → Transfer fails

Angle features:    Camera information is mathematically removed
                   → Source and target share the same geometric space
                   → A "bent finger" looks the same in any dataset
                   → Transfer succeeds
```

---

## 11. Key Experimental Results

### 11.1. Within-Domain (Table 3) — Angle Features Are More Discriminative

Best results per dataset (MLP encoder, 5-way 5-shot):

| Dataset | Raw | Angle | Raw+Angle | Best |
|---------|-----|-------|-----------|------|
| ASL | 94.9% | 88.4% | **95.4%** | raw_angle |
| LIBRAS | 81.2% | **94.1%** | 84.6% | angle |
| Arabic | 64.5% | **89.8%** | 71.0% | angle |
| Thai | 48.8% | **52.7%** | 51.9% | angle |

**Key observations**:
- Angle features dramatically outperform raw on LIBRAS (+12.9pp) and Arabic (+25.3pp)
- On ASL (largest, most uniform dataset), raw_angle wins because coordinate info helps
- Angle dominance grows as dataset quality/size decreases

### 11.2. Cross-Lingual Transfer (Table 4) — Angles Enable Frozen Transfer

ASL → Target (frozen MLP encoder, 5-shot):

| Target | Raw | Angle | Improvement |
|--------|-----|-------|-------------|
| LIBRAS | 86.5% | **95.0%** | +8.5pp |
| Arabic | 74.2% | **91.3%** | +17.1pp |
| Thai | 52.5% | 53.2% | +0.7pp |

Angle features achieve **95.0% on LIBRAS** with a frozen encoder pretrained on ASL —
with zero adaptation to LIBRAS.

### 11.3. Cross-Lingual Can Beat Within-Domain (Table 6)

Some transfer results **exceed** within-domain baselines:

| Transfer | Accuracy | Within-Domain | Gain |
|----------|----------|---------------|------|
| Arabic→LIBRAS | **97.1%** | 94.1% | +3.0pp |
| LIBRAS→Arabic | **91.7%** | 89.8% | +1.9pp |
| Arabic→Thai | **54.6%** | 52.7% | +1.9pp |

This is remarkable: pretraining on a **different language** helps more than
training on the target language alone, because the source provides richer
training signal.

### 11.4. Normalisation Ablation (Table 7) — Angles Are Truly Invariant

Removing wrist-centering and scale normalization:

| Representation | With Norm | Without Norm | Difference |
|---------------|-----------|--------------|-----------|
| Raw (LIBRAS) | 81.2% | 76.4% | **-4.8pp** |
| **Angle (LIBRAS)** | **94.1%** | **94.4%** | **-0.3pp** |
| Raw (Arabic) | 64.5% | 59.1% | **-5.4pp** |
| **Angle (Arabic)** | **89.8%** | **90.1%** | **-0.3pp** |

Raw coordinates lose 5pp without normalization. Angle features are **unchanged**
(within noise), empirically confirming the theoretical invariance proof.

### 11.5. The Cost of Few-Shot (Table 9)

Comparing 5-shot ProtoNet with full-data training:

| Method | ASL/raw | LIBRAS/angle | Arabic/angle |
|--------|---------|-------------|-------------|
| Full-data linear | 93.8% | 99.7% | - |
| 5-shot ProtoNet | 94.9% | 94.1% | 89.8% |
| Gap | +1.1pp | -5.6pp | - |

The 5-shot regime costs ~5.6pp on LIBRAS compared to full-data training.

---

## 12. Why It Works — Intuition

### The Prototype Quality Argument

```
With RAW features (camera-dependent):

    Support examples for "A":       Prototype for "A":
    ┌─────────────────────┐         ┌─────────────────┐
    │ Example 1 (cam pos A)│        │                 │
    │ Example 2 (cam pos B)│───────→│  NOISY centroid │──→ Wrong predictions
    │ Example 3 (cam pos C)│        │  (spread out)   │
    │ Example 4 (cam pos D)│        │                 │
    │ Example 5 (cam pos E)│        └─────────────────┘
    └─────────────────────┘


With ANGLE features (camera-INdependent):

    Support examples for "A":       Prototype for "A":
    ┌─────────────────────┐         ┌─────────────────┐
    │ Example 1 (same!)   │        │                 │
    │ Example 2 (same!)   │───────→│  TIGHT centroid │──→ Correct predictions
    │ Example 3 (same!)   │        │  (clustered)    │
    │ Example 4 (same!)   │        │                 │
    │ Example 5 (same!)   │        └─────────────────┘
    └─────────────────────┘
```

With only 5 examples, the prototype must be accurate. Angle features make it
accurate by removing all the "noise" from camera differences.

### The Information Compression Argument

```
63-D raw coordinates contain:
    ├── Hand shape information (what we want)
    ├── Camera viewpoint (noise)
    ├── Hand scale (noise)
    └── Hand position (noise)

20-D angle features contain:
    └── Hand shape information (what we want) — ONLY
```

By compressing 63 dimensions to 20, the paper throws away noise and keeps
only signal. This is a principled dimensionality reduction guided by physics
(invariance theory), not by statistics (PCA, autoencoders).

---

## 13. Limitations and Future Work

### Current Limitations

| Limitation | Detail |
|-----------|--------|
| **Static only** | Only works on single-frame fingerspelling poses, not dynamic signs or continuous signing |
| **Single hand** | Does not handle two-handed signs or signs involving body/face |
| **No bone lengths** | Angles discard absolute finger extension magnitude (how far a finger extends) |
| **Fingerspelling only** | Not tested on lexical signs (words/concepts) |
| **4 languages** | Broader coverage of non-fingerspelling sign lexicons needed |

### What Angles Cannot Distinguish

Some signs differ only in:
- **Finger extension magnitude** (how far the finger stretches) — angles are the same
  whether the finger is partially or fully extended
- **Hand size** — relevant for some signs but removed by scale invariance
- **Contact information** — which fingers touch each other

The Thai dataset analysis (Section 4.8) shows that confused pairs like classes
13↔14 and 28↔4 share similar angles but differ in these finer-grained features.

### Suggested Extensions

1. **Add bone-length ratios** — invariant to scale but captures extension differences
2. **Add contact features** — binary indicators for which fingertips touch
3. **Temporal modeling** — extend to dynamic signs using sequences of angle features
4. **Two-hand modeling** — add relative angle features between hands
5. **Multi-modal fusion** — use angles as a pose branch alongside RGB features

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| **Few-shot learning** | Learning to classify from very few examples (typically 1-5 per class) |
| **N-way K-shot** | An episode with N classes and K examples per class |
| **Prototypical Network** | A few-shot method that classifies by nearest-centroid matching |
| **Prototype** | The mean embedding of K support examples for one class |
| **Episode** | One mini-classification task: sample N classes, K+Q examples each |
| **Support set** | The K labelled examples used to build prototypes |
| **Query set** | The Q unlabelled examples to classify (labelled for evaluation) |
| **SO(3)** | The special orthogonal group — the group of all 3D rotation matrices |
| **Invariance** | A property that remains unchanged under a transformation |
| **Domain shift** | Distribution mismatch between training and test data |
| **Cross-lingual transfer** | Training on language A, testing on language B |
| **Frozen transfer** | Using the pretrained model without any adaptation |
| **Target-supervised** | Fine-tuning (partially) on the target language |
| **SupCon loss** | Supervised Contrastive Loss — pulls same-class embeddings together |
| **MediaPipe** | Google's framework for real-time hand/body tracking |
| **Keypoints** | Detected joint positions (landmarks) on the hand |
| **Fingerspelling** | Signing individual letters of an alphabet with hand shapes |
| **Kinematic chain** | A sequence of connected joints (like finger bones) |
| **Cosine annealing** | A learning rate schedule that follows a cosine curve down to 0 |
| **Early stopping** | Stopping training when validation performance stops improving |
| **Confidence interval** | A range likely containing the true mean (95% CI here) |

---

## Summary in One Paragraph

This paper solves the problem of recognizing sign language gestures across
different languages with minimal data by using a clever geometric trick:
instead of feeding raw 3D joint coordinates (which change with camera angle)
into a neural network, it computes the **angles between bones at each joint**,
which are mathematically guaranteed to be identical regardless of camera
position, rotation, or zoom. These 20 angles compress the hand pose into a
compact, camera-invariant descriptor that produces tighter class prototypes
in a Prototypical Network, enabling a model trained on American Sign Language
to recognize Brazilian, Arabic, or Thai signs with just 5 examples per class
— often outperforming models trained directly on those languages.
