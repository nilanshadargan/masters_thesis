"""Confusion-matrix plotting for the two classification dimensions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ..config import DATA_UNIQUENESS_CLASSES, UVP_CLASSES


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot a single confusion matrix; optionally save it to ``save_path``."""
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks, class_names, rotation=45, ha="right")
    ax.set_yticks(ticks, class_names)

    threshold = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            color="white" if cm[i, j] > threshold else "black",
        )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_result_matrices(
    results: dict[str, list],
    save_dir: str | Path | None = None,
) -> None:
    """Plot UVP and Data Uniqueness confusion matrices from a results dict.

    ``results`` is the dict returned by
    :func:`company_assessment.models.bert.cross_validate_model`
    (keys ``uvp_labels``/``uvp_preds`` and ``db_labels``/``db_preds``).
    """
    uvp_cm = confusion_matrix(results["uvp_labels"], results["uvp_preds"])
    db_cm = confusion_matrix(results["db_labels"], results["db_preds"])

    save_dir = Path(save_dir) if save_dir is not None else None
    plot_confusion_matrix(
        uvp_cm,
        UVP_CLASSES,
        "UVP Confusion Matrix",
        save_dir / "uvp_confusion_matrix.png" if save_dir else None,
    )
    plot_confusion_matrix(
        db_cm,
        DATA_UNIQUENESS_CLASSES,
        "Data Uniqueness Confusion Matrix",
        save_dir / "data_uniqueness_confusion_matrix.png" if save_dir else None,
    )


def plot_comparison_confusion_matrices(
    comparison: pd.DataFrame,
    title_prefix: str = "",
    save_path: str | Path | None = None,
) -> None:
    """Plot side-by-side UVP and Data Uniqueness confusion matrices (seaborn).

    ``comparison`` is a frame from
    :func:`company_assessment.evaluation.comparison.build_comparison_df`
    with ``*_true`` / ``*_predicted`` columns.
    """
    dimensions = [
        ("UVP", UVP_CLASSES, "UVP"),
        ("Data Uniqueness", DATA_UNIQUENESS_CLASSES, "Database Characteristics"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, (col, class_names, label) in zip(axes, dimensions):
        cm = confusion_matrix(
            comparison[f"{col}_true"],
            comparison[f"{col}_predicted"],
            labels=class_names,
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_title(f"{title_prefix}Confusion Matrix for {label}".strip())
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
