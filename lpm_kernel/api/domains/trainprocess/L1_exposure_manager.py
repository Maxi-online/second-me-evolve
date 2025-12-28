import os
import json
import pandas as pd
import numpy as np
from typing import Optional, Dict
from lpm_kernel.models.l1 import L1Bio, L1Shade, L1Cluster, L1ChunkTopic
from lpm_kernel.common.repository.database_session import DatabaseSession

# Output file mapping for each process step
output_files = {
    "extract_dimensional_topics": os.path.join(os.getcwd(), "resources/L2/data_pipeline/raw_data/topics.json"),
    "map_your_entity_network": os.path.join(os.getcwd(), "resources/L1/graphrag_indexing_output/subjective/entities.parquet"),
    "decode_preference_patterns": os.path.join(os.getcwd(), "resources/L2/data/preference.json"),
    "reinforce_identity": os.path.join(os.getcwd(), "resources/L2/data/selfqa.json"),
    "augment_content_retention": os.path.join(os.getcwd(), "resources/L2/data/diversity.json"),
}

def query_l1_version_data(version: int) -> dict:
    """
    Query L1 bio and shades for a given version and return as dict.
    """
    with DatabaseSession.session() as session:
            # Get all data for this version
            bio = session.query(L1Bio).filter(L1Bio.version == version).first()

            shades = session.query(L1Shade).filter(L1Shade.version == version).all()

            clusters = (
                session.query(L1Cluster).filter(L1Cluster.version == version).all()
            )

            chunk_topics = (
                session.query(L1ChunkTopic)
                .filter(L1ChunkTopic.version == version)
                .all()
            )

            if not bio:
                return jsonify(APIResponse.error(f"Version {version} not found"))

            # Build response data
            data = {
                "file_type": "json",
                "content": {
                    "version": version,
                    "bio": {
                        "content": bio.content,
                        "content_third_view": bio.content_third_view,
                        "summary": bio.summary,
                    "summary_third_view": bio.summary_third_view,
                    "shades": [
                        {
                            "name": s.name,
                            "aspect": s.aspect,
                            "icon": s.icon,
                            "desc_third_view": s.desc_third_view,
                            "content_third_view": s.content_third_view,
                            "desc_second_view": s.desc_second_view,
                            "content_second_view": s.content_second_view,
                        }
                        for s in shades
                    ],
                    },
                    "clusters": [
                        {
                            "cluster_id": c.cluster_id,
                            "memory_ids": c.memory_ids,
                            "cluster_center": c.cluster_center,
                        }
                        for c in clusters
                    ],
                    "chunk_topics": [
                        {"chunk_id": t.chunk_id, "topic": t.topic, "tags": t.tags}
                        for t in chunk_topics
                    ],
                }
            }
            return data

def read_file_content(file_path: str) -> Optional[Dict]:
    """Read content from a file based on its type."""
    try:
        if file_path.endswith(".json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                return {
                    "file_type": "json",
                    "content": content
                }
            except Exception as e:
                print(f"Error reading JSON file {file_path}: {str(e)}")
                return None
        elif file_path.endswith(".parquet"):
            return read_parquet_file(file_path)
        else:
            print(f"Unsupported file type for {file_path}")
            return None
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def read_parquet_file(file_path: str) -> Optional[Dict]:
    """
    Read a parquet file, convert numpy types for JSON serialization, and return file metadata and content.
    """
    try:
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        df = pd.read_parquet(file_path)
        # Remove columns named 'x' and 'y' if they exist
        df = df.drop(columns=[col for col in ['x', 'y'] if col in df.columns])
        df_dict = df.to_dict(orient='records')
        json_str = json.dumps(df_dict, cls=NumpyEncoder)
        records = json.loads(json_str)
        return {
            "file_type": "parquet",
            "rows": len(df),
            "columns": list(df.columns),
            "size_bytes": os.path.getsize(file_path),
            "content": records
        }
    except Exception as e:
        print(f"Error reading parquet file {file_path}: {str(e)}")
        return None
