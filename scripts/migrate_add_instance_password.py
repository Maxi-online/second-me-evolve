#!/usr/bin/env python
"""
Database Migration Script - Add instance_password column to tables with instance_id

This script adds the instance_password column to tables that have instance_id but not instance_password in the SQLite database.
It should be run once to update the database schema.
"""

import os
import sqlite3
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from lpm_kernel.configs.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_db_path():
    """Get the database path from environment or use default"""
    config = Config.from_env()
    db_path = config.get("SQLITE_DB_PATH", os.path.join(project_root, "data", "sqlite", "lpm.db"))
    return db_path

def migrate_database():
    """Add instance_password column to tables with instance_id"""
    db_path = get_db_path()
    
    logger.info(f"Using database at: {db_path}")
    
    # Check if database file exists
    if not os.path.exists(db_path):
        logger.error(f"Database file not found at {db_path}")
        return False
    
    # Default password to use
    default_password = "mindverse666"
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        logger.info(f"Tables in database: {[table[0] for table in tables]}")
        
        # Check for any table that might have instance_id but not instance_password
        for table_name in [table[0] for table in tables]:
            # Get columns for this table
            cursor.execute(f"PRAGMA table_info({table_name})")
            table_columns = cursor.fetchall()
            table_column_names = [column[1] for column in table_columns]
            
            # If table has instance_id but not instance_password
            if "instance_id" in table_column_names and "instance_password" not in table_column_names:
                logger.info(f"Table {table_name} has instance_id but not instance_password")
                logger.info(f"Adding instance_password column to {table_name} table")
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN instance_password VARCHAR(255) DEFAULT '{default_password}'")
                logger.info(f"Added instance_password column to {table_name} table with default value '{default_password}'")
                
                # Update existing rows to set the default password where instance_id is not null
                cursor.execute(f"UPDATE {table_name} SET instance_password = '{default_password}' WHERE instance_id IS NOT NULL AND instance_password IS NULL")
                updated_rows = cursor.rowcount
                logger.info(f"Updated {updated_rows} rows in {table_name} with default password")
        
        # Commit the changes
        conn.commit()
        logger.info("Migration completed successfully")
        
        # Close the connection
        conn.close()
        return True
        
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting database migration")
    success = migrate_database()
    
    if success:
        logger.info("Migration completed successfully")
        sys.exit(0)
    else:
        logger.error("Migration failed")
        sys.exit(1)
