import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pandas as pd

from dataset_loader import load_dataset_file, load_uploaded_dataset
from detector import compose_text, find_label_column, predict_post, train_and_compare
from storage import get_recent_predictions, initialize_database, save_prediction


def sample_jobs():
    rows = []
    for index in range(8):
        rows.append(
            {
                "title": f"Software engineer {index}",
                "company_profile": "Established technology company with a public office.",
                "description": "Full-time role building and maintaining customer software.",
                "requirements": "Relevant experience and a technical interview are required.",
                "fraudulent": 0,
            }
        )
        rows.append(
            {
                "title": f"Work from home agent {index}",
                "company_profile": "",
                "description": "Earn thousands daily, no interview. Pay a fee to start today.",
                "requirements": "No experience needed, send bank details immediately.",
                "fraudulent": 1,
            }
        )
    return pd.DataFrame(rows)


class DetectorTests(unittest.TestCase):
    def test_compose_text_ignores_empty_values(self):
        self.assertEqual(
            compose_text({"title": "Analyst", "description": None}), "title: Analyst"
        )

    def test_find_label_column_is_case_insensitive(self):
        self.assertEqual(
            find_label_column(pd.DataFrame(columns=["Title", "Fraudulent"])), "Fraudulent"
        )

    def test_training_compares_models_and_predicts(self):
        results = train_and_compare(sample_jobs())

        self.assertEqual(
            set(results.models),
            {"Logistic Regression", "Naive Bayes", "Soft Voting Ensemble"},
        )
        self.assertEqual(results.row_count, 16)
        self.assertEqual(results.test_row_count, 4)
        self.assertEqual(results.test_genuine_count + results.test_fraudulent_count, 4)
        self.assertIn(results.best_model, results.models)
        self.assertIn(results.recommended_model, results.models)
        self.assertEqual(
            set(results.metrics.columns), {"Model", "Accuracy", "Precision", "Recall", "F1"}
        )
        label, probability = predict_post(
            results.models[results.best_model],
            {
                "title": "Remote agent",
                "description": "Pay a fee and earn thousands today. No interview.",
            },
        )
        self.assertEqual(label, 1)
        self.assertTrue(0 <= probability <= 1)

    def test_training_rejects_single_class_data(self):
        with self.assertRaisesRegex(ValueError, "at least 4 genuine and 4 fraudulent"):
            train_and_compare(pd.DataFrame({"title": ["role"] * 8, "fraudulent": [0] * 8}))

    def test_sqlite_history_persists_only_screening_summary(self):
        with TemporaryDirectory() as directory:
            database_path = Path(directory) / "test.db"
            initialize_database(database_path)
            save_prediction("Support specialist", 1, 0.82, "Naive Bayes", database_path)

            history = get_recent_predictions(database_path=database_path)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["Title"], "Support specialist")
        self.assertEqual(history[0]["Result"], "Potential risk")
        self.assertAlmostEqual(history[0]["Score"], 0.82)
        self.assertEqual(history[0]["Model"], "Naive Bayes")
        self.assertNotIn("description", history[0])

    def test_dataset_loader_reads_csv_from_zip(self):
        csv_bytes = sample_jobs().to_csv(index=False).encode("utf-8")
        with TemporaryDirectory() as directory:
            archive_path = Path(directory) / "kaggle.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("fake_job_postings.csv", csv_bytes)

            from_path = load_dataset_file(archive_path)
            from_upload = load_uploaded_dataset(archive_path.name, archive_path.read_bytes())

        self.assertEqual(len(from_path), 16)
        self.assertEqual(from_path["fraudulent"].sum(), 8)
        self.assertTrue(from_upload.equals(from_path))