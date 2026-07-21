"""Classification models: fine-tuned BERT and the GPT few-shot classifier."""

from .bert import (
    CompanyDataset,
    MultiLabelBERT,
    cross_validate_model,
    prepare_companies,
    tokenize_text,
)
from .chatgpt import (
    build_few_shot_messages,
    classify_company,
    get_client,
    parse_classification,
)

__all__ = [
    # BERT
    "MultiLabelBERT",
    "CompanyDataset",
    "prepare_companies",
    "tokenize_text",
    "cross_validate_model",
    # ChatGPT
    "get_client",
    "build_few_shot_messages",
    "classify_company",
    "parse_classification",
]
