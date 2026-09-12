"""
Evaluation metrics for few-shot sign language recognition.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy.

    Args:
        logits: Prediction logits of shape ``(N, C)``.
        targets: Ground-truth labels of shape ``(N,)``.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def few_shot_accuracy_with_ci(
    accuracies: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute mean accuracy with 95 % confidence interval.

    Args:
        accuracies: Per-episode accuracy values.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (mean, ci_half_width).
    """
    arr = np.array(accuracies)
    mean = arr.mean()
    std = arr.std(ddof=1)
    # 95 % CI  ≈  1.96 * std / sqrt(n)
    z = 1.96 if confidence == 0.95 else 2.576
    ci = z * std / np.sqrt(len(arr))
    return float(mean), float(ci)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute a confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional list of label names.

    Returns:
        Confusion matrix as a NumPy array.
    """
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: str = "results/plots/confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a confusion matrix.

    Args:
        cm: Confusion matrix array.
        class_names: Optional class label names.
        save_path: Path to save the figure.
        title: Figure title.
    """
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=6)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str = "results/plots/tsne.png",
    title: str = "t-SNE of Embeddings",
    perplexity: int = 30,
) -> None:
    """Generate and save a t-SNE visualisation of embeddings.

    Args:
        embeddings: Array of shape ``(N, D)``.
        labels: Integer labels of shape ``(N,)``.
        save_path: Path to save the figure.
        title: Figure title.
        perplexity: t-SNE perplexity.
    """
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i)], label=str(lbl), s=8, alpha=0.7,
        )
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=6, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def cross_domain_accuracy_drop(
    source_acc: float,
    target_acc: float,
) -> Dict[str, float]:
    """Compute accuracy drop when transferring across domains.

    Args:
        source_acc: Accuracy on source domain.
        target_acc: Accuracy on target domain.

    Returns:
        Dictionary with source, target, and drop values.
    """
    return {
        "source_accuracy": source_acc,
        "target_accuracy": target_acc,
        "accuracy_drop": source_acc - target_acc,
        "relative_drop_pct": (source_acc - target_acc) / max(source_acc, 1e-8) * 100,
    }
