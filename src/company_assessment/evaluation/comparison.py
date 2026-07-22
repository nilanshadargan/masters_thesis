"""Turn raw model predictions into true-vs-predicted comparison tables.

Used by the results analysis (thesis §4) to score each GPT model against the
manually annotated ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import DATA_UNIQUENESS_CLASSES, PROCESSED_DIR, UVP_CLASSES

DIMENSIONS = ("UVP", "Data Uniqueness")
CORRECTIONS_DIR = PROCESSED_DIR / "corrections"


def apply_label_corrections(
    df: pd.DataFrame, model_key: str, corrections_dir: Path = CORRECTIONS_DIR
) -> pd.DataFrame:
    """Apply the manual per-model label corrections stored in ``corrections/``.

    Corrections are hand-curated fixes to noisy raw model output (e.g. off-vocab
    labels like "Moderately Unique"), keyed by row index. Returns a corrected copy.
    ``model_key`` is one of ``gpt3_5``, ``gpt4o``, ``gpt-4-turbo``.
    """
    df = df.copy()
    corrections = json.loads((corrections_dir / f"{model_key}.json").read_text())
    for fix in corrections:
        df.at[fix["index"], fix["column"]] = fix["value"]
    return df


def build_comparison_df(
    true_labels: pd.DataFrame,
    predicted: pd.DataFrame,
    dimensions: tuple[str, ...] = DIMENSIONS,
    id_col: str = "company_id",
) -> pd.DataFrame:
    """Merge true and predicted labels into one ``*_true`` / ``*_predicted`` frame.

    Predictions where *every* dimension is NaN are dropped (the model failed to
    classify that company); any remaining gaps are filled with ``"N/A"``.
    """
    cols = [id_col, *dimensions]
    pred = predicted[cols].copy()
    pred = pred[~pred[list(dimensions)].isna().all(axis=1)]

    comparison = true_labels[cols].merge(pred, on=id_col, suffixes=("_true", "_predicted"))
    return comparison.fillna("N/A")


# Class label order per dimension, for confusion matrices and per-class recall.
CLASS_LABELS: dict[str, list[str]] = {
    "UVP": UVP_CLASSES,
    "Data Uniqueness": DATA_UNIQUENESS_CLASSES,
}


def per_class_recall(comparison: pd.DataFrame, dimension: str) -> dict[str, float]:
    """Recall for each class of ``dimension`` (thesis Tables 1a / 1b).

    A class with no true examples scores 0.0. This is the per-class accuracy
    reported in the thesis appendix.
    """
    true = comparison[f"{dimension}_true"]
    pred = comparison[f"{dimension}_predicted"]
    recalls = {}
    for cls in CLASS_LABELS[dimension]:
        support = (true == cls).sum()
        correct = ((true == cls) & (pred == cls)).sum()
        recalls[cls] = correct / support if support else 0.0
    return recalls


def dimension_accuracy(comparison: pd.DataFrame, dimension: str) -> float:
    """Macro-averaged per-class recall for one dimension.

    This is the "accuracy" reported in thesis Table 3: the unweighted mean of the
    per-class recalls (:func:`per_class_recall`), NOT the overall fraction correct.
    Macro-averaging is the appropriate choice here because the classes are highly
    imbalanced. Use :func:`dimension_micro_accuracy` for the raw fraction correct.
    """
    recalls = per_class_recall(comparison, dimension)
    return sum(recalls.values()) / len(recalls)


def dimension_micro_accuracy(comparison: pd.DataFrame, dimension: str) -> float:
    """Overall fraction of companies classified correctly for one dimension."""
    from sklearn.metrics import accuracy_score

    return accuracy_score(comparison[f"{dimension}_true"], comparison[f"{dimension}_predicted"])


def accuracy_report(
    comparison: pd.DataFrame, dimensions: tuple[str, ...] = DIMENSIONS
) -> dict[str, float]:
    """Per-dimension and overall accuracy as a dict (thesis Table 3 convention).

    Each dimension's score is the macro-averaged per-class recall; ``overall`` is
    the mean across dimensions.
    """
    report = {dim: dimension_accuracy(comparison, dim) for dim in dimensions}
    report["overall"] = sum(report.values()) / len(dimensions)
    return report
