#!/usr/bin/env python
"""
Database Migration Runner

This script runs database migrations using the migration manager.
It should be executed whenever the database schema needs to be updated.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from lpm_kernel.configs.config import Config
from lpm_kernel.database.migration_manager import MigrationManager

from lpm_kernel.common.logging import logger

def get_db_path():
    """Get the database path from environment or use default"""
    config = Config.from_env()
    db_path = config.get("SQLITE_DB_PATH", os.path.join(project_root, "data", "sqlite", "lpm.db"))
    return db_path

def run_migrations():
    """Run all pending database migrations"""
    db_path = get_db_path()
    
    # logger.info(f"Using database at: {db_path}")
    
    # Check if database file exists
    if not os.path.exists(db_path):
        # logger.error(f"Database file not found at {db_path}")
        return False
    
    try:
        # Initialize migration manager
        migrations_dir = os.path.join(project_root, "lpm_kernel", "database", "migrations")
        manager = MigrationManager(db_path)
        
        # Apply migrations
        applied = manager.apply_migrations(migrations_dir)
        
        # if applied:
        #     logger.info(f"Successfully applied {len(applied)} migrations")
        # else:
        #     logger.info("No new migrations to apply")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during migrations: {e}")
        return False

def create_migration(description):
    """Create a new migration file"""
    db_path = get_db_path()
    migrations_dir = os.path.join(project_root, "lpm_kernel", "database", "migrations")
    
    manager = MigrationManager(db_path)
    filepath = manager.create_migration(description, migrations_dir)
    
    # logger.info(f"Created new migration at: {filepath}")
    return filepath

if __name__ == "__main__":
    # logger.info("Starting database migration")
    
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        if len(sys.argv) > 2:
            description = sys.argv[2]
            create_migration(description)
        else:
            logger.error("Missing migration description")
            print("Usage: python run_migrations.py create 'Add new table'")
            sys.exit(1)
    else:
        success = run_migrations()
        
        if success:
            # logger.info("Migration completed successfully")
            sys.exit(0)
        else:
            logger.error("Migration failed")
            sys.exit(1)
