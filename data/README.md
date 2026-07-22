# Data

This project uses a three-stage data layout. Only **`processed/`** is tracked in
git — it holds the manual annotations and model results that are the research
output. `raw/` and `interim/` are git-ignored because they are large and/or
regenerable; obtain them as described below.

```
data/
├── raw/         # external source data        (git-ignored — download)
├── interim/     # scraped / filtered data      (git-ignored — regenerate)
└── processed/   # annotated data + results     (tracked in git)
```

## `raw/` — external source data (git-ignored)

| File | Description |
|------|-------------|
| `2023-07-13-yc-companies.csv` | Y Combinator Directory dump (~4,000 companies). Primary base dataset. |
| `2023-02-27-yc-companies.csv` | Earlier YC Directory snapshot. |

**How to obtain:** download from the Kaggle *Y Combinator Directory* dataset and
place the CSVs here:
<https://www.kaggle.com/datasets/miguelcorraljr/y-combinator-directory>

## `interim/` — scraped & filtered data (git-ignored)

Produced by `notebooks/02_dataset_creation.ipynb`. Regenerable, but note that
scraping hits live company websites, so re-running is slow and non-deterministic
(sites change over time).

| File | Description |
|------|-------------|
| `scraped_ai.csv` | AI-tagged YC companies enriched with a `Scraped_Info` column from their landing pages. |
| `filtered_data.csv` | `scraped_ai` filtered to rows with ≥100 chars of scraped text (244 companies). |
| `filtered_data.legacy.csv` | An earlier variant of the filtered dataset, kept for provenance. |

## `processed/` — annotated data & results (tracked)

The irreplaceable outputs: manual annotations and model classifications.

| File | Description |
|------|-------------|
| `data_companies.csv` | Annotated evaluation set (111 companies) with manual `UVP` and `Data Uniqueness` labels. Input to the BERT model. |
| `examples_chatgpt.csv` | The 8 few-shot example companies shown to the GPT models. |
| `final_chatgpt.csv` | The 103 companies to be classified by the GPT models (few-shot input). |
| `final.csv` | Consolidated evaluation dataset. |
| `final_classified_gpt3turbo.csv` | GPT-3.5-turbo-0125 predictions. |
| `final_classified_gpt4o.csv` | GPT-4o predictions. |
| `final_classified_gpt-4-turbo.csv` | GPT-4-turbo predictions (best model). |
| `newprompt_final_classified_gpt-4-turbo.csv` | GPT-4-turbo predictions with the revised prompt. |
| `comparison_df_gpt3_5.csv` | True vs. predicted labels for GPT-3.5-turbo (used for metrics/confusion matrices). |
| `comparison_df_gpt4.csv` | True vs. predicted labels for GPT-4-turbo. |
| `comparison_df_gpt4o.csv` | True vs. predicted labels for GPT-4o. |

## Classification dimensions

Each company is labelled on two dimensions, four classes each:

**Unique Value Proposition (UVP)** — predicted longevity of the offering:
`Already obsolete` · `Ok for 1-2 years` · `Ok for 3-5 years` · `Ok for 5-8 years`

**Data Uniqueness** — accessibility/exclusivity of the company's data:
`Large/Difficult to Obtain` · `Quantity questionable` · `Not Special/Publicly Available` · `N/A`
