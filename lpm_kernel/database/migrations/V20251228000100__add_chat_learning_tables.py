"""
Migration: Add chat learning tables (experiences, feedback, preferences)
Version: 20251228000100
"""

description = "Add chat learning tables (experiences, feedback, preferences)"


def upgrade(conn):
    """
    Apply the migration

    Args:
        conn: SQLite connection object
    """
    cursor = conn.cursor()

    # chat_experiences
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_experiences (
            id VARCHAR(64) PRIMARY KEY,
            source VARCHAR(50) NOT NULL DEFAULT 'kernel2',
            model VARCHAR(200),
            prompt_messages TEXT NOT NULL, -- JSON stored as TEXT
            temperature INTEGER,           -- temperature * 1000
            max_tokens INTEGER,
            seed INTEGER,
            completion TEXT,
            finish_reason VARCHAR(50),
            meta_data TEXT NOT NULL DEFAULT '{}', -- JSON stored as TEXT
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_experiences_created_at ON chat_experiences(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_experiences_source ON chat_experiences(source)")

    # chat_feedback
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_feedback (
            id VARCHAR(64) PRIMARY KEY,
            experience_id VARCHAR(64) NOT NULL,
            rating INTEGER NOT NULL DEFAULT 0,
            comment TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experience_id) REFERENCES chat_experiences(id)
        );
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_feedback_experience_id ON chat_feedback(experience_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_feedback_created_at ON chat_feedback(created_at)")

    # chat_preferences
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_preferences (
            id VARCHAR(64) PRIMARY KEY,
            source VARCHAR(50) NOT NULL DEFAULT 'kernel2',
            model VARCHAR(200),
            prompt_messages TEXT NOT NULL, -- JSON stored as TEXT
            chosen TEXT NOT NULL,
            rejected TEXT NOT NULL,
            meta_data TEXT NOT NULL DEFAULT '{}', -- JSON stored as TEXT
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_preferences_created_at ON chat_preferences(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_preferences_source ON chat_preferences(source)")


def downgrade(conn):
    """
    Revert the migration

    Args:
        conn: SQLite connection object
    """
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS chat_feedback")
    cursor.execute("DROP TABLE IF EXISTS chat_preferences")
    cursor.execute("DROP TABLE IF EXISTS chat_experiences")

