from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from contextlib import contextmanager
import logging
from lpm_kernel.configs.config import Config
import os

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class DatabaseSession:
    _instance = None
    _engine = None
    _session_factory = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls.initialize()
        return cls._instance

    @classmethod
    def initialize(cls):
        """Initialize database engine and session factory"""
        if not cls._engine:
            try:
                config = Config.from_env()
                db_config = config.database.to_dict()

                # Ensure database directory exists
                db_dir = os.path.dirname(db_config['db_file'])
                if not os.path.exists(db_dir):
                    os.makedirs(db_dir)
                
                # Build SQLite connection URL
                db_url = f"sqlite:///{db_config['db_file']}"

                cls._engine = create_engine(
                    db_url,
                    echo=False,
                    pool_pre_ping=True,
                    pool_recycle=db_config['pool_recycle'],
                    pool_size=db_config['maxsize'],
                    max_overflow=20,
                )
                cls._session_factory = sessionmaker(bind=cls._engine)
                logger.info("SQLite database engine and session factory initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database: {str(e)}")
                raise

    @classmethod
    @contextmanager
    def session(cls):
        """Get database session"""
        if not cls._session_factory:
            cls.initialize()

        session = cls._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @classmethod
    def close(cls):
        """Close database engine - should only be called when shutting down the application"""
        if cls._engine:
            cls._engine.dispose()
            cls._engine = None
            cls._session_factory = None
            logger.info("Database engine closed")
