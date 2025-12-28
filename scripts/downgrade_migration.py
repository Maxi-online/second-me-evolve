#!/usr/bin/env python
"""
Database Migration Downgrade Tool

Usage:
    python downgrade_migration.py [version]

Arguments:
    version: Optional, version to downgrade to (inclusive).
             If not provided, all migrations will be downgraded.
"""

import os
import sys
import sqlite3
import importlib.util
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import migration manager
from lpm_kernel.database.migration_manager import MigrationManager
from lpm_kernel.configs.config import Config

def main():
    # Get database path
    config = Config.from_env()
    db_path = config.database.db_file
    
    print(f"Database path: {db_path}")
    
    # Create migration manager
    manager = MigrationManager(db_path)
    
    # Get applied migrations
    applied = manager.get_applied_migrations()
    print(f"Applied migrations: {', '.join(applied) if applied else 'none'}")
    
    if not applied:
        print("No migrations to downgrade")
        return
    
    # Check command line arguments
    if len(sys.argv) > 1:
        target_version = sys.argv[1]
        if target_version not in applied:
            print(f"Error: Version {target_version} is not applied, cannot downgrade to this version")
            return
        
        print(f"Downgrading to version {target_version}...")
        downgraded = manager.downgrade_to_version(target_version)
    else:
        print("Downgrading all migrations...")
        downgraded = manager.downgrade_to_version()
    
    if downgraded:
        print(f"Successfully downgraded migrations: {', '.join(downgraded)}")
    else:
        print("No migrations were downgraded")

if __name__ == "__main__":
    main()
