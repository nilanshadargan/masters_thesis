"""Central configuration: filesystem paths, label mappings, and shared constants.

Everything path- or label-related lives here so the notebooks and scripts never
hard-code a string. Paths are resolved relative to the repository root, so code
works regardless of the current working directory.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------
# config.py lives at src/company_assessment/config.py, so the repo root is 3
# levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Frequently used files
YC_COMPANIES_CSV = RAW_DIR / "2023-07-13-yc-companies.csv"
SCRAPED_AI_CSV = INTERIM_DIR / "scraped_ai.csv"
FILTERED_DATA_CSV = INTERIM_DIR / "filtered_data.csv"
ANNOTATED_CSV = PROCESSED_DIR / "data_companies.csv"
EXAMPLES_CSV = PROCESSED_DIR / "examples_chatgpt.csv"
FINAL_CHATGPT_CSV = PROCESSED_DIR / "final_chatgpt.csv"

# --- Label mappings ------------------------------------------------------
UVP_LABELS: dict[str, int] = {
    "Already obsolete": 0,
    "Ok for 1-2 years": 1,
    "Ok for 3-5 years": 2,
    "Ok for 5-8 years": 3,
}

DATA_UNIQUENESS_LABELS: dict[str, int] = {
    "Large/Difficult to Obtain": 0,
    "Quantity questionable": 1,
    "Not Special/Publicly Available": 2,
    "N/A": 3,
}

# Ordered class names (index == encoded label) for plotting / reports.
UVP_CLASSES: list[str] = list(UVP_LABELS)
DATA_UNIQUENESS_CLASSES: list[str] = list(DATA_UNIQUENESS_LABELS)

# Text columns fed to the models.
TEXT_COLUMNS: list[str] = ["short_description", "long_description", "Scraped_Info"]

# --- Model constants -----------------------------------------------------
BERT_MODEL_NAME = "bert-base-uncased"
MAX_TOKENS = 512
RANDOM_STATE = 42


def ensure_dirs() -> None:
    """Create the data/output directories if they do not yet exist."""
    for path in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
