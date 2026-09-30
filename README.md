# Postmark: Fake Job Post Screening

A small Streamlit application for screening job-post text with TF-IDF features and three classifiers: Logistic Regression, Multinomial Naive Bayes, and a soft-voting ensemble.

Screening history is stored locally in `postmark.db` using SQLite. It records only the job title, model result, score, model name, and timestamp; full job-post text is not saved.

## Run it

Python 3.8 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. To run the focused checks:

```powershell
python -m unittest discover -s tests -v
```

## Use a labeled dataset

At startup, the app looks for `fake_job_postings.csv` beside `app.py`, then `~/Downloads/archive (4).zip`, and trains on it when found. Set `POSTMARK_DATASET_PATH` to use a different local CSV or ZIP path. If no local dataset is found, the app trains on a small **synthetic walkthrough corpus**; its metrics are not meaningful evidence of production performance.

In the sidebar, upload the original CSV or its ZIP archive and select **Train from uploaded dataset**. The dataset's `fraudulent` column is supported (`0` for genuine and `1` for fraudulent). Text features such as `title`, `location`, `company_profile`, `description`, `requirements`, and `benefits` are combined for classification. A stratified 25% hold-out split is used to compare the three models; the selected models are then refit on the full labeled upload.

Supported label-column names are `fraudulent`, `is_fraudulent`, `label`, `class`, and `target`. Values can be `0`/`1`, `fake`/`real`, `fraudulent`/`legitimate`, or equivalent supported yes/no values. At least four examples of each class are required.

## Limitations

This is a text-only screening aid. Scores are model outputs, not calibrated probabilities. Text can be incomplete, legitimate language can look suspicious, and scam wording changes. Do not use the result as the sole basis for decisions; verify employers through independently located official contact details. Never pay an application fee or share passwords or bank logins with a recruiter.