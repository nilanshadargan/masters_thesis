"""Automation of Company Assessment for the Investment Process.

Reusable library code behind the MSc thesis: classifying AI startups along the
Unique Value Proposition and Data Uniqueness dimensions with BERT and GPT models.

Submodules:
    config      - paths, label mappings, constants
    data        - dataset creation (scraping / filtering)
    models      - BERT and ChatGPT classifiers
    evaluation  - metrics and confusion-matrix plotting
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import config  # noqa: F401
