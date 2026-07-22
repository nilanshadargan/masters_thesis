"""Few-shot startup classification with the OpenAI Chat Completions API.

The model is primed with eight labelled example companies (system + alternating
user/assistant turns), then asked to classify new companies along UVP and Data
Uniqueness (thesis section 3.6 and Appendix B).

The API key is read from the ``OPENAI_API_KEY`` environment variable (loaded from
a local ``.env`` file if present) — never hard-code it.
"""

from __future__ import annotations

import os
import re

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a helpful assistant trained to classify company profiles based on "
    "their descriptions along two dimensions: Unique Value Proposition (UVP) and "
    "Database Uniqueness."
)

DEFAULT_MODEL = "gpt-4-turbo"

# Regex to pull the two dimensions out of a model reply such as
# "UVP - Ok for 1-2 years, Data Uniqueness - Large/Difficult to Obtain."
_CLASSIFICATION_PATTERN = r"UVP - (.*?)(?:,|$) Data Uniqueness - (.*?)(?:\.$|$)"


def get_client(api_key: str | None = None) -> OpenAI:
    """Return an OpenAI client, reading the key from the environment/.env.

    Raises a clear error if no key is available.
    """
    load_dotenv()
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return OpenAI(api_key=api_key)


def _describe(row: pd.Series) -> str:
    """Render a company's text columns into the user-turn prompt string."""
    return (
        "Classify the company with these descriptions: "
        f"Short description: '{row.get('short_description')}', "
        f"Long description: '{row.get('long_description')}', "
        f"Scraped Info: '{row.get('Scraped_Info')}'."
    )


def build_few_shot_messages(examples: pd.DataFrame) -> list[dict]:
    """Build the system + few-shot message list from labelled examples.

    Each example contributes a user turn (its descriptions) and an assistant turn
    (its gold ``UVP``/``Data Uniqueness`` labels).
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for _, row in examples.iterrows():
        messages.append({"role": "user", "content": _describe(row)})
        messages.append(
            {
                "role": "assistant",
                "content": (f"UVP - {row['UVP']}, " f"Data Uniqueness - {row['Data Uniqueness']}."),
            }
        )
    return messages


def classify_company(
    client: OpenAI,
    company: pd.Series,
    few_shot_messages: list[dict],
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Classify one company and return the raw model reply."""
    messages = few_shot_messages + [{"role": "user", "content": _describe(company)}]
    response = client.chat.completions.create(model=model_name, messages=messages)
    return response.choices[0].message.content


def parse_classification(reply: str) -> tuple[str | None, str | None]:
    """Extract ``(uvp, data_uniqueness)`` from a model reply, or ``(None, None)``."""
    match = re.search(_CLASSIFICATION_PATTERN, reply or "")
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def classify_dataframe(
    client: OpenAI,
    companies: pd.DataFrame,
    examples: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Classify every company, returning a copy with prediction columns added."""
    few_shot = build_few_shot_messages(examples)
    companies = companies.copy()
    total = len(companies)

    replies = []
    for i, (_, row) in enumerate(companies.iterrows(), 1):
        print(f"Classifying company {i} of {total}")
        replies.append(classify_company(client, row, few_shot, model_name))

    companies["classification"] = replies
    parsed = companies["classification"].apply(parse_classification)
    companies["UVP_predicted"] = [p[0] for p in parsed]
    companies["Data Uniqueness_predicted"] = [p[1] for p in parsed]
    return companies
