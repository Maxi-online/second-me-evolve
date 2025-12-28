"""
Migration: Add judge model fields to user_llm_configs table
Version: 20251228000200
"""

description = "Add judge model fields to user_llm_configs table"


def upgrade(conn):
    """
    Apply the migration

    Args:
        conn: SQLite connection object
    """
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(user_llm_configs)")
    columns = [row[1] for row in cursor.fetchall()]

    if "judge_model_name" not in columns:
        cursor.execute("ALTER TABLE user_llm_configs ADD COLUMN judge_model_name VARCHAR(200)")
        print("Added judge_model_name column to user_llm_configs table")

    if "judge_endpoint" not in columns:
        cursor.execute("ALTER TABLE user_llm_configs ADD COLUMN judge_endpoint VARCHAR(200)")
        print("Added judge_endpoint column to user_llm_configs table")

    if "judge_api_key" not in columns:
        cursor.execute("ALTER TABLE user_llm_configs ADD COLUMN judge_api_key VARCHAR(200)")
        print("Added judge_api_key column to user_llm_configs table")


def downgrade(conn):
    """
    Revert the migration (SQLite doesn't support drop column directly).
    We keep downgrade as a no-op to avoid destructive table rebuild for a non-critical change.
    """
    print("Downgrade skipped: SQLite does not support dropping columns safely in-place.")

