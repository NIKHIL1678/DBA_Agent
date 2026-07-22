"""
Runs Schema.sql directly against MySQL using pymysql — no mysql CLI needed.

Usage (from anywhere, path below is relative to this file's location):
    python -m Data.Run_Schema

Reads DB_HOST / DB_PORT / DB_USER / DB_PASSWORD from .env (via python-dotenv).
Does NOT require DB_NAME to already exist — the schema file itself contains
CREATE DATABASE IF NOT EXISTS, so this connects without selecting a database
first.
"""

import os
import re
from pathlib import Path

import pymysql
from dotenv import load_dotenv

load_dotenv()

SCHEMA_FILE = Path(__file__).parent / "Schema.sql"


def split_sql_statements(sql_text: str) -> list[str]:
    """
    Splits a .sql file's contents into individual executable statements.
    Strips full-line '--' comments first, then splits on ';'.
    Good enough for straightforward DDL files without stored procedures
    or semicolons embedded inside string literals.
    """
    # Remove full-line comments (lines starting with -- after stripping whitespace)
    lines = sql_text.splitlines()
    cleaned_lines = [line for line in lines if not line.strip().startswith("--")]
    cleaned_sql = "\n".join(cleaned_lines)

    # Split on statement-terminating semicolons
    statements = [stmt.strip() for stmt in cleaned_sql.split(";")]
    return [stmt for stmt in statements if stmt]


def run_schema():
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Schema file not found at {SCHEMA_FILE}")

    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")

    print(f"Connecting to MySQL at {host}:{port} as {user} (no database selected yet)...")

    connection = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        autocommit=True,
    )

    try:
        sql_text = SCHEMA_FILE.read_text(encoding="utf-8")
        statements = split_sql_statements(sql_text)

        print(f"Executing {len(statements)} statements from {SCHEMA_FILE.name}...\n")

        with connection.cursor() as cursor:
            for i, statement in enumerate(statements, start=1):
                preview = re.sub(r"\s+", " ", statement)[:80]
                print(f"[{i}/{len(statements)}] {preview}...")
                cursor.execute(statement)

        print("\nSchema applied successfully.")

    finally:
        connection.close()


if __name__ == "__main__":
    run_schema()