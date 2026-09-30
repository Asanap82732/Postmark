"""SQLite storage for a minimal, privacy-conscious screening history."""

import sqlite3
from contextlib import closing
from pathlib import Path


DATABASE_PATH = Path(__file__).with_name("postmark.db")


def initialize_database(database_path=DATABASE_PATH):
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_title TEXT NOT NULL,
                is_fraudulent INTEGER NOT NULL CHECK (is_fraudulent IN (0, 1)),
                fraud_score REAL NOT NULL CHECK (fraud_score >= 0 AND fraud_score <= 1),
                model_name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_prediction(job_title, is_fraudulent, fraud_score, model_name, database_path=DATABASE_PATH):
    title = str(job_title or "").strip() or "Untitled post"
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO predictions (job_title, is_fraudulent, fraud_score, model_name)
            VALUES (?, ?, ?, ?)
            """,
            (title, int(is_fraudulent), float(fraud_score), str(model_name)),
        )


def get_recent_predictions(limit=20, database_path=DATABASE_PATH):
    safe_limit = max(1, min(int(limit), 100))
    with closing(sqlite3.connect(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT job_title AS Title,
                   CASE is_fraudulent WHEN 1 THEN 'Potential risk' ELSE 'No strong signal' END AS Result,
                   fraud_score AS Score,
                   model_name AS Model,
                   created_at AS Screened
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return [dict(row) for row in rows]