"""Build the evaluation dataset from the Y Combinator Directory dump.

Pipeline (see thesis section 3.1 "Dataset Creation"):

1. Load the raw YC companies CSV.
2. Keep companies tagged "Artificial Intelligence" and drop duplicates.
3. Scrape each company's landing page into a ``Scraped_Info`` column.
4. Filter out companies with too little scraped text.

Scraping hits live websites, so results are slow and non-deterministic.
"""

from __future__ import annotations

import ast

import pandas as pd
import requests
from bs4 import BeautifulSoup

FAILED_MARKER = "Failed to retrieve"
MIN_SCRAPED_CHARS = 100


def _parse_tags(value: object) -> object:
    """Parse a tags cell that may be a stringified list into a real list."""
    if isinstance(value, str) and value.startswith("["):
        return ast.literal_eval(value)
    return value


def scrape_website(url: str, company_name: str, timeout: int = 10) -> str:
    """Fetch ``url`` and return its visible text joined by spaces.

    Returns :data:`FAILED_MARKER` if the request fails.
    """
    try:
        response = requests.get(url, timeout=timeout)
        soup = BeautifulSoup(response.content, "html.parser")
        print(f"Scraping website for {company_name}")
        return " ".join(soup.stripped_strings)
    except requests.RequestException:
        return FAILED_MARKER


def filter_ai_companies(ycd: pd.DataFrame) -> pd.DataFrame:
    """Return YC companies tagged 'Artificial Intelligence', de-duplicated."""
    ycd = ycd.copy()
    ycd["tags"] = ycd["tags"].apply(_parse_tags)
    is_ai = ycd["tags"].apply(
        lambda tags: "Artificial Intelligence" in tags if isinstance(tags, list) else False
    )
    return ycd[is_ai].drop_duplicates(subset=["company_name"])


def build_scraped_ai_dataset(ycd: pd.DataFrame) -> pd.DataFrame:
    """Filter to AI companies and enrich each with a ``Scraped_Info`` column.

    Rows whose scrape failed or is empty are dropped.
    """
    ai_companies = filter_ai_companies(ycd)
    ai_companies["Scraped_Info"] = ai_companies.apply(
        lambda row: scrape_website(row["website"], row["company_name"]), axis=1
    )
    scraped = ai_companies[
        (ai_companies["Scraped_Info"] != FAILED_MARKER) & (ai_companies["Scraped_Info"].notna())
    ]
    return scraped.drop_duplicates(subset=["company_name"])


def filter_by_scraped_length(
    scraped: pd.DataFrame, min_chars: int = MIN_SCRAPED_CHARS
) -> pd.DataFrame:
    """Keep only rows with at least ``min_chars`` of scraped text."""
    return scraped[scraped["Scraped_Info"].str.len() >= min_chars]
