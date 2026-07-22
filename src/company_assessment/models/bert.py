"""Fine-tuned BERT for multi-label startup classification.

A single ``bert-base-uncased`` encoder feeds two classification heads, one per
dimension (UVP and Data Uniqueness). Training uses 5-fold cross-validation with
class weights to counter class imbalance (thesis section 3.4).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

from ..config import (
    BERT_MODEL_NAME,
    DATA_UNIQUENESS_LABELS,
    MAX_TOKENS,
    RANDOM_STATE,
    TEXT_COLUMNS,
    UVP_LABELS,
)


def _clean_text(text: str) -> str:
    """Collapse runs of whitespace and strip."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize_text(
    texts: list[str], tokenizer: BertTokenizer, max_length: int = MAX_TOKENS
) -> tuple[list, list]:
    """Batch-encode ``texts`` into input ids and attention masks."""
    encoded = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    )
    return encoded["input_ids"], encoded["attention_mask"]


def prepare_companies(
    companies: pd.DataFrame, tokenizer: BertTokenizer | None = None
) -> pd.DataFrame:
    """Clean text, encode labels, and tokenise.

    Adds ``text``, ``uvp_label``, ``db_label``, ``input_ids`` and
    ``attention_masks`` columns. Rows must already carry ``UVP`` and
    ``Data Uniqueness`` labels.
    """
    companies = companies.copy()
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    companies["text"] = companies[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1).apply(_clean_text)
    companies["uvp_label"] = companies["UVP"].map(UVP_LABELS)
    companies["db_label"] = companies["Data Uniqueness"].map(DATA_UNIQUENESS_LABELS)

    input_ids, attention_masks = tokenize_text(companies["text"].tolist(), tokenizer)
    companies["input_ids"] = input_ids
    companies["attention_masks"] = attention_masks
    return companies.reset_index(drop=True)


class CompanyDataset(Dataset):
    """Torch dataset yielding tokenised text and both label sets."""

    def __init__(self, input_ids, attention_masks, uvp_labels, db_labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.uvp_labels = uvp_labels
        self.db_labels = db_labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "uvp_label": torch.tensor(self.uvp_labels[idx], dtype=torch.long),
            "db_label": torch.tensor(self.db_labels[idx], dtype=torch.long),
        }


class MultiLabelBERT(torch.nn.Module):
    """BERT encoder with two linear heads for two classification tasks."""

    def __init__(self, bert_model_name: str, num_labels1: int, num_labels2: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = torch.nn.Dropout(0.4)
        self.classifier1 = torch.nn.Linear(self.bert.config.hidden_size, num_labels1)
        self.classifier2 = torch.nn.Linear(self.bert.config.hidden_size, num_labels2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled_output = self.dropout(outputs.pooler_output)
        return self.classifier1(pooled_output), self.classifier2(pooled_output)


def cross_validate_model(
    companies: pd.DataFrame,
    num_splits: int = 5,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-2,
) -> dict[str, list]:
    """Run k-fold cross-validation and collect out-of-fold predictions.

    ``companies`` must have been passed through :func:`prepare_companies`.
    Returns a dict with ``uvp_preds``/``uvp_labels`` and ``db_preds``/``db_labels``.
    """
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_uvp_preds: list = []
    all_uvp_labels: list = []
    all_db_preds: list = []
    all_db_labels: list = []

    class_weights_uvp = torch.tensor(
        compute_class_weight(
            "balanced",
            classes=np.unique(companies["uvp_label"]),
            y=companies["uvp_label"],
        ),
        dtype=torch.float,
    ).to(device)
    class_weights_db = torch.tensor(
        compute_class_weight(
            "balanced",
            classes=np.unique(companies["db_label"]),
            y=companies["db_label"],
        ),
        dtype=torch.float,
    ).to(device)

    for fold, (train_index, val_index) in enumerate(kf.split(companies["input_ids"]), 1):
        print(f"Fold {fold}")
        train_dataset = CompanyDataset(
            [companies["input_ids"][i] for i in train_index],
            [companies["attention_masks"][i] for i in train_index],
            [companies["uvp_label"][i] for i in train_index],
            [companies["db_label"][i] for i in train_index],
        )
        val_dataset = CompanyDataset(
            [companies["input_ids"][i] for i in val_index],
            [companies["attention_masks"][i] for i in val_index],
            [companies["uvp_label"][i] for i in val_index],
            [companies["db_label"][i] for i in val_index],
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = MultiLabelBERT(
            BERT_MODEL_NAME,
            num_labels1=len(set(companies["uvp_label"])),
            num_labels2=len(set(companies["db_label"])),
        ).to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(epochs):
            model.train()
            total_loss, total_correct, total_samples = 0.0, 0, 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                uvp_labels = batch["uvp_label"].to(device)
                db_labels = batch["db_label"].to(device)

                uvp_logits, db_logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(
                    uvp_logits, uvp_labels, weight=class_weights_uvp
                ) + F.cross_entropy(db_logits, db_labels, weight=class_weights_db)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_correct += (uvp_logits.argmax(1) == uvp_labels).sum().item()
                total_correct += (db_logits.argmax(1) == db_labels).sum().item()
                total_samples += uvp_labels.size(0) + db_labels.size(0)

            print(
                f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, "
                f"Accuracy: {total_correct / total_samples:.4f}"
            )

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                uvp_logits, db_logits = model(input_ids, attention_mask)
                all_uvp_preds.extend(uvp_logits.argmax(1).cpu().numpy())
                all_uvp_labels.extend(batch["uvp_label"].numpy())
                all_db_preds.extend(db_logits.argmax(1).cpu().numpy())
                all_db_labels.extend(batch["db_label"].numpy())

    return {
        "uvp_preds": all_uvp_preds,
        "uvp_labels": all_uvp_labels,
        "db_preds": all_db_preds,
        "db_labels": all_db_labels,
    }
