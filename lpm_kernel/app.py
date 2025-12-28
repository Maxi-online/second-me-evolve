from flask import Flask, request
from .common.repository.database_session import DatabaseSession, Base
from .common.logging import logger
from .api import init_routes
from .api.file_server.handler import FileServerHandler
from .database.migration_manager import MigrationManager
import os
import atexit
import subprocess
from lpm_kernel.api.domains.kernel2.services.reinforce_scheduler import start_auto_reinforce_dpo
from lpm_kernel.api.domains.kernel2.services.clickhouse_store import clickhouse_store


def create_app():
    app = Flask(__name__)

    # Initialize database connection
    try:
        DatabaseSession.initialize()
        logger.info("Database connection initialized successfully")
    
    except Exception as e:
        logger.error(f"Failed to initialize database connection: {str(e)}")
        raise

        # Add CORS support

    @app.after_request
    def after_request(response):
        # Allow all origins in development environment
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add(
            "Access-Control-Allow-Headers", "Content-Type,Authorization"
        )
        response.headers.add(
            "Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS"
        )
        return response

    # Create file server handler
    file_handler = FileServerHandler(
        os.path.join(os.getenv("APP_ROOT", "/app"), "resources", "raw_content")
    )

    @app.route("/raw_content/", defaults={"path": ""})
    @app.route("/raw_content/<path:path>")
    def serve_content(path=""):
        return file_handler.handle_request(path, request.path)

    # Register all routes
    init_routes(app)

    # ClickHouse is REQUIRED for RL logs; ensure schema exists at startup.
    try:
        clickhouse_store.ensure_schema()
    except Exception as e:
        logger.error(f"ClickHouse is required but not available: {e}")
        raise

    # Optional: auto reinforcement learning loop (disabled by default)
    # Enable via: AUTO_REINFORCE_DPO_ENABLED=1
    try:
        start_auto_reinforce_dpo()
    except Exception as e:
        logger.warning(f"Auto reinforce scheduler failed to start: {e}")

    # Clean up database connection only when the application shuts down
    @app.teardown_appcontext
    def cleanup_db(exception):
        pass

    return app


app = create_app()


@atexit.register
def cleanup():
    DatabaseSession.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
