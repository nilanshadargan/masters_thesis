"""Text-processing helpers for the exploratory data analysis (notebook 01).

Reading pitch decks / spreadsheets / documents, cleaning text, and simple
similarity/frequency analysis. The document readers depend on the optional
``eda`` extras (``PyPDF2``, ``python-docx``): install with ``pip install -e .[eda]``.
"""

from __future__ import annotations

import re
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_excel_file(file_path, columns=None):
    """Read specified columns from an Excel file (or an error string)."""
    import pandas as pd

    try:
        return pd.read_excel(file_path, usecols=columns)
    except Exception as exc:  # noqa: BLE001 - EDA helper: surface the error inline
        return f"Error reading {file_path}: {exc}"


def read_word_file(file_path) -> str:
    """Read all paragraph text from a Word document (or an error string)."""
    import docx

    try:
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as exc:  # noqa: BLE001
        return f"Error reading {file_path}: {exc}"


def read_pdf_file(file_path) -> str:
    """Extract text from a PDF using PyPDF2 >= 3.0 (or an error string)."""
    import PyPDF2

    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            return "\n".join(page.extract_text() for page in reader.pages)
    except Exception as exc:  # noqa: BLE001
        return f"Error reading {file_path}: {exc}"


def clean_text(input_text: str) -> str:
    """Replace newlines with spaces and strip punctuation."""
    cleaned = input_text.replace("\n", " ")
    return re.sub(r"[^\w\s]", "", cleaned)


def count_words(text: str) -> int:
    """Count whitespace-separated tokens."""
    return len(text.split())


def top_10_frequent_words(text: str, remove_stopwords: bool = False) -> list[tuple[str, int]]:
    """Return the 10 most frequent words, optionally excluding English stop words."""
    words = re.findall(r"\b\w+\b", text.lower())
    if remove_stopwords:
        sw = set(stopwords.words("english"))
        words = [word for word in words if word not in sw]
    return Counter(words).most_common(10)


def remove_stop_words(text: str) -> str:
    """Remove English stop words from ``text``."""
    stop_words = set(stopwords.words("english"))
    return " ".join(word for word in word_tokenize(text) if word.lower() not in stop_words)


def calculate_cosine_similarity(text: str, list_of_dimensions: list[str]):
    """TF-IDF cosine similarity between ``text`` and each dimension description."""
    documents = list_of_dimensions + [text]
    tfidf_matrix = TfidfVectorizer().fit_transform(documents)
    return cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
