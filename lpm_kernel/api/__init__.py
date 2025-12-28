from flask import Flask
from .domains.health.routes import health_bp
from .domains.documents.routes import document_bp
from .domains.kernel.routes import kernel_bp
from .domains.kernel2.routes_l2 import kernel2_bp
from .domains.loads.routes import loads_bp
from .domains.memories.routes import memories_bp
from .domains.kernel2.routes.role_routes import role_bp
from .domains.trainprocess import trainprocess_bp
from .domains.upload.routes import upload_bp
from .domains.kernel2.routes_talk import talk_bp
from .domains.user_llm_config.routes import user_llm_config_bp
from .domains.space.space_routes import space_bp

def init_routes(app: Flask):
    """Initialize all route blueprints"""
    
    app.register_blueprint(health_bp)
    app.register_blueprint(document_bp)
    app.register_blueprint(kernel_bp)
    app.register_blueprint(kernel2_bp)
    app.register_blueprint(loads_bp)
    app.register_blueprint(memories_bp)
    app.register_blueprint(role_bp)
    app.register_blueprint(trainprocess_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(space_bp)
    app.register_blueprint(talk_bp)
    app.register_blueprint(user_llm_config_bp)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    # Disable response buffering
    app.config['RESPONSE_BUFFERING'] = False


# To maintain compatibility with existing code
routes = init_routes
