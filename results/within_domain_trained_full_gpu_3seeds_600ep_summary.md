# Trained Within-Domain Three-Seed Summary

Cells show mean accuracy +/- seed standard deviation, in percent.

## asl_alphabet

| Encoder | Representation | 1-shot | 3-shot | 5-shot |
|---------|----------------|--------|--------|--------|
| mlp | raw | 97.92 +/- 0.08 | 98.82 +/- 0.13 | 99.04 +/- 0.12 |
| mlp | angle | 95.03 +/- 0.59 | 97.55 +/- 0.43 | 97.81 +/- 0.36 |
| mlp | raw_angle | 97.78 +/- 0.26 | 98.69 +/- 0.02 | 98.80 +/- 0.10 |
| transformer | raw | 96.70 +/- 0.68 | 98.62 +/- 0.24 | 98.71 +/- 0.17 |
| transformer | angle | 94.34 +/- 0.56 | 96.88 +/- 0.31 | 97.44 +/- 0.19 |
| transformer | raw_angle | 98.05 +/- 0.12 | 98.69 +/- 0.05 | 98.85 +/- 0.03 |

## arabic_sign_alphabet

| Encoder | Representation | 1-shot | 3-shot | 5-shot |
|---------|----------------|--------|--------|--------|
| mlp | raw | 94.98 +/- 0.07 | 97.59 +/- 0.17 | 98.00 +/- 0.22 |
| mlp | angle | 93.24 +/- 0.24 | 95.95 +/- 0.17 | 96.31 +/- 0.16 |
| mlp | raw_angle | 95.94 +/- 0.25 | 98.11 +/- 0.09 | 98.38 +/- 0.04 |
| transformer | raw | 93.31 +/- 0.14 | 96.89 +/- 0.41 | 97.11 +/- 0.60 |
| transformer | angle | 93.00 +/- 0.49 | 96.07 +/- 0.22 | 96.38 +/- 0.08 |
| transformer | raw_angle | 96.52 +/- 0.26 | 98.36 +/- 0.02 | 98.59 +/- 0.01 |

## libras_alphabet

| Encoder | Representation | 1-shot | 3-shot | 5-shot |
|---------|----------------|--------|--------|--------|
| mlp | raw | 98.22 +/- 0.40 | 99.18 +/- 0.16 | 99.32 +/- 0.10 |
| mlp | angle | 98.57 +/- 0.13 | 99.23 +/- 0.08 | 99.41 +/- 0.10 |
| mlp | raw_angle | 98.81 +/- 0.37 | 99.58 +/- 0.10 | 99.72 +/- 0.04 |
| transformer | raw | 97.47 +/- 0.40 | 98.13 +/- 0.31 | 98.33 +/- 0.16 |
| transformer | angle | 98.15 +/- 0.17 | 99.03 +/- 0.04 | 99.24 +/- 0.07 |
| transformer | raw_angle | 99.33 +/- 0.23 | 99.78 +/- 0.07 | 99.85 +/- 0.07 |

## thai_fingerspelling

| Encoder | Representation | 1-shot | 3-shot | 5-shot |
|---------|----------------|--------|--------|--------|
| mlp | raw | 75.39 +/- 1.24 | 83.99 +/- 0.96 | 85.62 +/- 0.71 |
| mlp | angle | 73.11 +/- 0.51 | 81.72 +/- 0.93 | 83.40 +/- 0.35 |
| mlp | raw_angle | 78.18 +/- 0.97 | 86.14 +/- 0.62 | 87.25 +/- 0.52 |
| transformer | raw | 71.33 +/- 0.82 | 80.38 +/- 1.13 | 82.09 +/- 1.17 |
| transformer | angle | 69.04 +/- 0.21 | 79.27 +/- 0.62 | 81.83 +/- 0.54 |
| transformer | raw_angle | 79.07 +/- 0.63 | 86.65 +/- 0.71 | 88.05 +/- 0.43 |
