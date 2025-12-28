
from .health.routes import health_bp
from .documents.routes import document_bp
from .kernel.routes import kernel_bp
from .kernel2.routes_l2 import kernel2_bp
from .kernel2.routes_talk import talk_bp
from .space.space_routes import space_bp
from .user_llm_config.routes import user_llm_config_bp
from .memories.routes import memories_bp
# from .config.routes import config_bp
from .trainprocess.routes import trainprocess_bp
from .upload.routes import upload_bp
from .loads.routes import loads_bp



__all__ = ["health_bp", "document_bp", "kernel_bp", "kernel2_bp", "talk_bp", "space_bp", "user_llm_config_bp", "memories_bp", "trainprocess_bp", "upload_bp", "loads_bp"]