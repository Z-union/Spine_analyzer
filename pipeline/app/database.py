# database.py
import sqlite3
from datetime import datetime

DATABASE_PATH = "pipeline_results.db"

def init_db():
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_results (
                id TEXT PRIMARY KEY,
                study_id TEXT NOT NULL,
                results TEXT NOT NULL,
                processed_at TEXT NOT NULL,
                report_path TEXT
            )
        ''')
        conn.commit()

def save_processing_result(processing_id: str, study_id: str, results_json: str, report_path: str):
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO processing_results (id, study_id, results, processed_at, report_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (processing_id, study_id, results_json, datetime.utcnow().isoformat(), report_path))
        conn.commit()

def get_result_by_study_id(study_id: str):
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.row_factory = sqlite3.Row  # To return dicts
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM processing_results WHERE study_id = ?', (study_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_result_by_processing_id(processing_id: str):
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM processing_results WHERE id = ?', (processing_id,))
        row = cursor.fetchone()
        return dict(row) if row else None