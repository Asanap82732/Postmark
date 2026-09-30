"""Text-based fake job posting classification utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


TEXT_COLUMNS = (
    "title",
    "location",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
)
LABEL_NAMES = ("fraudulent", "is_fraudulent", "label", "class", "target")
MODEL_NAMES = ("Logistic Regression", "Naive Bayes", "Soft Voting Ensemble")
POSITIVE_LABELS = {"1", "true", "yes", "fake", "fraud", "fraudulent", "scam"}
NEGATIVE_LABELS = {"0", "false", "no", "real", "legitimate", "genuine", "not fraudulent"}


@dataclass
class TrainingResults:
    models: dict[str, Pipeline]
    metrics: pd.DataFrame
    best_model: str
    recommended_model: str
    row_count: int
    test_row_count: int
    test_genuine_count: int
    test_fraudulent_count: int


def compose_text(record: dict[str, Any]) -> str:
    parts = []
    for column in TEXT_COLUMNS:
        value = record.get(column)
        if value is None or pd.isna(value):
            continue
        cleaned = str(value).strip()
        if cleaned:
            parts.append(f"{column.replace('_', ' ')}: {cleaned}")
    return " ".join(parts)


def find_label_column(frame: pd.DataFrame) -> str:
    columns = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in LABEL_NAMES:
        if candidate in columns:
            return columns[candidate]
    raise ValueError(
        "Could not find a label column. Expected one of: " + ", ".join(LABEL_NAMES)
    )


def _normalize_labels(values: pd.Series) -> pd.Series:
    normalized = []
    for value in values:
        if pd.isna(value):
            raise ValueError("The label column contains empty values.")
        label = str(value).strip().lower()
        if label in POSITIVE_LABELS:
            normalized.append(1)
        elif label in NEGATIVE_LABELS:
            normalized.append(0)
        else:
            raise ValueError(
                f"Unsupported label {value!r}. Use 0/1, fake/real, or fraudulent/legitimate."
            )
    return pd.Series(normalized, index=values.index, dtype="int64")


def make_estimator() -> VotingClassifier:
    logistic = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    naive_bayes = MultinomialNB(alpha=0.5)
    return VotingClassifier(
        estimators=[("logistic", logistic), ("naive_bayes", naive_bayes)],
        voting="soft",
        weights=[2, 1],
    )


def _make_estimators() -> dict[str, Any]:
    logistic = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    naive_bayes = MultinomialNB(alpha=0.5)
    return {
        "Logistic Regression": logistic,
        "Naive Bayes": naive_bayes,
        "Soft Voting Ensemble": make_estimator(),
    }


def train_and_compare(frame: pd.DataFrame) -> TrainingResults:
    label_column = find_label_column(frame)
    usable = frame.dropna(subset=[label_column]).copy()
    if usable.empty:
        raise ValueError("The uploaded CSV has no labeled job postings.")

    labels = _normalize_labels(usable[label_column])
    counts = labels.value_counts()
    if counts.get(0, 0) < 4 or counts.get(1, 0) < 4:
        raise ValueError("Training needs at least 4 genuine and 4 fraudulent labeled posts.")

    texts = usable.apply(lambda row: compose_text(row.to_dict()), axis=1)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
        min_df=1,
        sublinear_tf=True,
    )
    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)

    metric_rows = []
    for name, estimator in _make_estimators().items():
        estimator.fit(train_vectors, train_labels)
        predictions = estimator.predict(test_vectors)
        metric_rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(test_labels, predictions),
                "Precision": precision_score(test_labels, predictions, zero_division=0),
                "Recall": recall_score(test_labels, predictions, zero_division=0),
                "F1": f1_score(test_labels, predictions, zero_division=0),
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values("F1", ascending=False).reset_index(drop=True)

    full_vectorizer = clone(vectorizer)
    full_vectors = full_vectorizer.fit_transform(texts)
    models = {}
    for name, estimator in _make_estimators().items():
        estimator.fit(full_vectors, labels)
        models[name] = Pipeline([("tfidf", full_vectorizer), ("classifier", estimator)])

    return TrainingResults(
        models=models,
        metrics=metrics,
        best_model=str(metrics.iloc[0]["Model"]),
        recommended_model=str(
            metrics.sort_values(["Recall", "Precision"], ascending=False).iloc[0]["Model"]
        ),
        row_count=len(usable),
        test_row_count=len(test_labels),
        test_genuine_count=int((test_labels == 0).sum()),
        test_fraudulent_count=int((test_labels == 1).sum()),
    )


def predict_post(model: Pipeline, record: dict[str, Any]) -> tuple[int, float]:
    text = compose_text(record)
    if not text:
        raise ValueError("Add a title or some job-posting text before analyzing.")
    probabilities = model.predict_proba([text])[0]
    fraud_index = list(model.classes_).index(1)
    probability = float(probabilities[fraud_index])
    return int(probability >= 0.5), probability