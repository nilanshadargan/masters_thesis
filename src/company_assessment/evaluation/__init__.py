"""Evaluation: baseline metrics, comparison tables, and confusion-matrix plotting."""

from .comparison import (
    accuracy_report,
    apply_label_corrections,
    build_comparison_df,
    dimension_accuracy,
    dimension_micro_accuracy,
    per_class_recall,
)
from .metrics import (
    majority_class_accuracy,
    per_class_accuracy,
    random_predictor_accuracy,
)
from .plots import (
    plot_comparison_confusion_matrices,
    plot_confusion_matrix,
    plot_result_matrices,
)

__all__ = [
    # metrics
    "majority_class_accuracy",
    "random_predictor_accuracy",
    "per_class_accuracy",
    # comparison
    "apply_label_corrections",
    "build_comparison_df",
    "per_class_recall",
    "dimension_accuracy",
    "dimension_micro_accuracy",
    "accuracy_report",
    # plots
    "plot_confusion_matrix",
    "plot_result_matrices",
    "plot_comparison_confusion_matrices",
]
