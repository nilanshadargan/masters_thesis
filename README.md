# Automation of Company Assessment for the Investment Process

Classifying AI startups along multiple investment-relevant dimensions with
**BERT** and **GPT** models, in a **few-shot** setting.

This repository accompanies the MSc thesis *"Automation of Company Assessment for
the Investment Process"* (University of Amsterdam, 2024), carried out with
[Azimut Zero](https://azimutzero.nl). The goal is to help automate venture-capital
due diligence by classifying startups from their textual profiles.

**Research question:** *To what extent can we classify AI startup companies along
multiple dimensions for the investment process in a few-shot manner?*

---

## What it does

Each company is classified along two dimensions, four classes each:

| Dimension | Classes |
|-----------|---------|
| **Unique Value Proposition (UVP)** — predicted longevity of the offering | `Already obsolete` · `Ok for 1-2 years` · `Ok for 3-5 years` · `Ok for 5-8 years` |
| **Data Uniqueness** — accessibility/exclusivity of the company's data | `Large/Difficult to Obtain` · `Quantity questionable` · `Not Special/Publicly Available` · `N/A` |

Four models are compared: fine-tuned **BERT** (`bert-base-uncased`, 110M params) and
three **GPT** models via the OpenAI API (`gpt-3.5-turbo-0125`, `gpt-4o`,
`gpt-4-turbo`) prompted with eight labelled few-shot examples.

## Pipeline

```
YC Directory dump (~4,000 companies)          data/raw/
        │  filter to "Artificial Intelligence" tag  -> 319 companies
        v
   scrape landing pages (Scraped_Info)         data/interim/
        │  keep rows with >=100 chars scraped   -> 244 companies
        v
   manual annotation (UVP + Data Uniqueness)   data/processed/
        │  111 companies, author + expert review
        v
 +--------------+---------------+
 v                              v
BERT (5-fold CV, class weights)   GPT (8-shot: 8 examples -> classify 103)
 │                              │
 v                              v
        per-class accuracy + confusion matrices    outputs/figures/
```

Notebook order mirrors the pipeline:

| Notebook | Stage |
|----------|-------|
| [`01_exploratory_data_analysis.ipynb`](notebooks/01_exploratory_data_analysis.ipynb) | Exploratory data analysis |
| [`02_dataset_creation.ipynb`](notebooks/02_dataset_creation.ipynb) | Scrape + filter the YC dataset |
| [`03_bert_classification.ipynb`](notebooks/03_bert_classification.ipynb) | Fine-tune & cross-validate BERT |
| [`04_chatgpt_classification.ipynb`](notebooks/04_chatgpt_classification.ipynb) | Few-shot classification with GPT |
| [`05_results_analysis.ipynb`](notebooks/05_results_analysis.ipynb) | Metrics, comparisons, confusion matrices |

## Repository structure

```
.
├── src/company_assessment/     # reusable library code
│   ├── config.py               # paths, label mappings, constants
│   ├── data/scraping.py        # dataset creation (scrape + filter)
│   ├── models/bert.py          # MultiLabelBERT + cross-validation
│   ├── models/chatgpt.py       # few-shot GPT classification
│   └── evaluation/             # baseline metrics + confusion-matrix plots
├── notebooks/                  # narrative pipeline (import from the package)
├── data/                       # raw / interim / processed  (see data/README.md)
├── outputs/figures/            # confusion matrices
├── requirements.txt
└── pyproject.toml
```

## Setup

Requires Python >= 3.10.

```bash
# 1. Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install the package and its dependencies
pip install -e .          # or: pip install -r requirements.txt

# 3. Configure your OpenAI key (only needed for notebook 04)
cp .env.example .env      # then edit .env and add your key
```

### Data

Only `data/processed/` (annotations + model results) ships with the repo. The raw
YC dumps and interim scraped files are git-ignored — see
[`data/README.md`](data/README.md) for how to download / regenerate them.

## Usage

Run the notebooks in order, or import the library directly:

```python
import pandas as pd
from company_assessment import config
from company_assessment.models.bert import prepare_companies, cross_validate_model
from company_assessment.evaluation import plot_result_matrices

companies = pd.read_csv(config.ANNOTATED_CSV, na_values=[""], keep_default_na=False)
companies["Data Uniqueness"] = companies["Data Uniqueness"].fillna("N/A")
companies = companies.dropna(subset=["UVP"]).reset_index(drop=True)

companies = prepare_companies(companies)
results = cross_validate_model(companies)
plot_result_matrices(results, save_dir=config.FIGURES_DIR)
```

```python
from company_assessment.models.chatgpt import get_client, classify_dataframe
# get_client() reads OPENAI_API_KEY from your .env / environment
```

## Results

Accuracy across both dimensions (range 0–1). Here "accuracy" is the
**macro-averaged per-class recall** — the unweighted mean of the four per-class
recalls — which is the appropriate metric for these imbalanced classes and the
one reported in the thesis. `notebooks/05_results_analysis.ipynb` reproduces
these numbers (and the per-class breakdown) from the committed predictions.

| Model | UVP | Data Uniqueness | Overall |
|-------|----:|----------------:|--------:|
| Fine-tuned BERT | 0.250 | 0.312 | 0.281 |
| GPT web interface | 0.283 | 0.334 | 0.308 |
| gpt-3.5-turbo-0125 | 0.227 | 0.287 | 0.257 |
| gpt-4o | 0.307 | 0.364 | 0.336 |
| **gpt-4-turbo** | **0.357** | **0.358** | **0.358** |
| *Random baseline* | 0.310 | 0.370 | 0.340 |
| *Majority-class baseline* | 0.450 | 0.500 | 0.475 |

**Takeaways.** GPT-4-turbo was the strongest model overall. Larger-parameter GPT
models captured more nuance than BERT, which was hampered by sensitivity to subtle
context changes and overfitting on the small dataset. All models still trailed the
majority-class baseline, underscoring the difficulty of the task under limited
annotated data and class imbalance — the key limitations discussed in the thesis.

## Citation

> Dargan, N. (2024). *Automation of Company Assessment for the Investment Process*.
> MSc thesis, Master Information Studies (Data Science), University of Amsterdam.

## License

Released under the MIT License (see [`LICENSE`](LICENSE)).
