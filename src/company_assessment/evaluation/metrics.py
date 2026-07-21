"""Baseline and per-class accuracy metrics (thesis section 3.7)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix


def majority_class_accuracy(labels: Sequence) -> float:
    """Accuracy of always predicting the most frequent class."""
    counts = Counter(labels)
    return counts.most_common(1)[0][1] / len(labels)


def random_predictor_accuracy(labels: Sequence) -> float:
    """Expected accuracy of a class-frequency-weighted random predictor."""
    counts = Counter(labels)
    total = sum(counts.values())
    return sum((count / total) ** 2 for count in counts.values())


def per_class_accuracy(y_true: Sequence, y_pred: Sequence, num_classes: int) -> list[float]:
    """Per-class recall (diagonal of the row-normalised confusion matrix)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1)
        diag = np.diag(cm)
        acc = np.divide(diag, row_sums, out=np.zeros_like(diag, dtype=float), where=row_sums != 0)
    return acc.tolist()
