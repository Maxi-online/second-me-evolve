"""
Migration: Add thinking model fields to user_llm_configs table
Version: 20250420221300
"""

description = "Add thinking model fields to user_llm_configs table"

def upgrade(conn):
    """
    Apply the migration
    
    Args:
        conn: SQLite connection object
    """
    cursor = conn.cursor()
    
    # Check if thinking_model_name column already exists in user_llm_configs table
    cursor.execute("PRAGMA table_info(user_llm_configs)")
    columns = [row[1] for row in cursor.fetchall()]
    
    # Add thinking model fields if they don't exist
    if 'thinking_model_name' not in columns:
        cursor.execute("ALTER TABLE user_llm_configs ADD COLUMN thinking_model_name VARCHAR(200)")
        print("Added thinking_model_name column to user_llm_configs table")
    
    if 'thinking_endpoint' not in columns:
        cursor.execute("ALTER TABLE user_llm_configs ADD COLUMN thinking_endpoint VARCHAR(200)")
        print("Added thinking_endpoint column to user_llm_configs table")
    
    if 'thinking_api_key' not in columns:
        cursor.execute("ALTER TABLE user_llm_configs ADD COLUMN thinking_api_key VARCHAR(200)")
        print("Added thinking_api_key column to user_llm_configs table")
    
    # No need to commit, the migration manager handles transactions

def downgrade(conn):
    """
    Revert the migration
    
    Args:
        conn: SQLite connection object
    """
    cursor = conn.cursor()
    
    # SQLite doesn't support dropping columns directly
    # We need to create a new table without the thinking model fields, copy the data, and replace the old table
    
    # Create a temporary table without thinking model fields
    cursor.execute("""
    CREATE TABLE user_llm_configs_temp (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider_type VARCHAR(50) NOT NULL DEFAULT 'openai',
        key VARCHAR(200),
        chat_endpoint VARCHAR(200),
        chat_api_key VARCHAR(200),
        chat_model_name VARCHAR(200),
        embedding_endpoint VARCHAR(200),
        embedding_api_key VARCHAR(200),
        embedding_model_name VARCHAR(200),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Copy data from the original table to the temporary table
    cursor.execute("""
    INSERT INTO user_llm_configs_temp (
        id, provider_type, key, 
        chat_endpoint, chat_api_key, chat_model_name,
        embedding_endpoint, embedding_api_key, embedding_model_name,
        created_at, updated_at
    )
    SELECT 
        id, provider_type, key, 
        chat_endpoint, chat_api_key, chat_model_name,
        embedding_endpoint, embedding_api_key, embedding_model_name,
        created_at, updated_at
    FROM user_llm_configs
    """)
    
    # Drop the original table
    cursor.execute("DROP TABLE user_llm_configs")
    
    # Rename the temporary table to the original table name
    cursor.execute("ALTER TABLE user_llm_configs_temp RENAME TO user_llm_configs")
    
    # Recreate the index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_llm_configs_created_at ON user_llm_configs(created_at)")
    
    print("Removed thinking model fields from user_llm_configs table")
    
    # No need to commit, the migration manager handles transactions
