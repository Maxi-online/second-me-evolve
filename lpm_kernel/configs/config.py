import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv


@dataclass
class DatabaseConfig:
    """Database configuration"""

    db_file: str
    pool_size: int = 5
    pool_recycle: int = 3600

    def to_dict(self) -> Dict:
        return {
            "db_file": self.db_file,
            "maxsize": self.pool_size,
            "pool_recycle": self.pool_recycle,
        }


@dataclass
class Config:
    """Application configuration class"""

    _instance = None

    # Core configuration uses fixed data structure
    app_name: str
    version: str
    word: str
    database: DatabaseConfig

    # Store other dynamic configurations
    _extra_config: Dict = None  # Add default value

    def __post_init__(self):
        """Execute after initialization, ensure _extra_config is initialized"""
        if self._extra_config is None:
            self._extra_config = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def from_env(cls, env_file: str = None) -> "Config":
        """Create configuration instance from environment variables"""
        if cls._instance is not None:
            return cls._instance

        # Load .env file
        env_path = Path(env_file) if env_file else Path(__file__).parent.parent.parent / ".env"

        if env_path.exists():
            load_dotenv(env_path)
        else:
            raise FileNotFoundError(f"Config file not found: {env_path}")

        # Get base directory configuration
        base_dir = os.getenv("BASE_DIR", "/app")

        instance = cls(
            app_name=os.getenv("APP_NAME", "simple-env-sdk"),
            version=os.getenv("APP_VERSION", "0.1.0"),
            word=os.getenv("APP_WORD", "hello world"),
            database=DatabaseConfig(
                db_file=os.getenv("DB_FILE", "data/sqlite/lpm.db"),
                pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
                pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
            ),
        )

        # Load all other environment variables to _extra_config
        for key, value in os.environ.items():
            if not hasattr(instance, key.lower()):  # Avoid overriding core configuration
                # If it's a directory configuration, ensure using the correct base directory
                if key.endswith("_DIR") and not os.path.isabs(value):
                    value = os.path.join(base_dir, value)
                instance._extra_config[key] = value

        # Vector store settings
        instance.CHROMA_PERSIST_DIRECTORY = os.getenv(
            "CHROMA_PERSIST_DIRECTORY", os.path.join(base_dir, "data/chroma_db")
        )
        instance.CHROMA_COLLECTION_NAME = os.getenv(
            "CHROMA_COLLECTION_NAME", "documents"
        )

        # Service URLs
        local_app_port = os.getenv("LOCAL_APP_PORT")
        
        # Kernel2 service URL
        instance.KERNEL2_SERVICE_URL = os.getenv(
            "KERNEL2_SERVICE_URL", f"http://127.0.0.1:{local_app_port}"
        )
        
        # Registry service URL
        instance.REGISTRY_SERVICE_URL = os.getenv(
            "REGISTRY_SERVICE_URL"
        )

        return instance

    @property
    def db_config(self) -> Dict:
        """Get database configuration dictionary"""
        return self.database.to_dict()

    def get(self, key: str, default: str = None) -> str:
        """Get configuration value
        First look in core configuration, if not found then look in dynamic configuration
        """
        # First check if it's a core configuration attribute
        if hasattr(self, key):
            return getattr(self, key)
        # Then look in dynamic configuration
        return self._extra_config.get(key, default)
