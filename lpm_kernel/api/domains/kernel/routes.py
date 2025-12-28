import logging
import os
from datetime import datetime

import numpy as np
from flask import Blueprint, jsonify

from lpm_kernel.L1.bio import Bio
# from lpm_kernel.L1.l1_repository import L1Repository
from lpm_kernel.L1.l1_generator import L1Generator
from lpm_kernel.L1.serializers import NotesStorage, NoteSerializer
from lpm_kernel.L1.utils import save_true_topics
from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.common.repository.database_session import DatabaseSession
from lpm_kernel.kernel.chunk_service import ChunkService
from lpm_kernel.kernel.l1.l1_manager import (
    generate_l1_from_l0,
    generate_and_store_status_bio,
    get_latest_status_bio,
    extract_notes_from_documents,
    document_service,
)
from lpm_kernel.kernel.note_service import NoteService
from lpm_kernel.models.l1 import (
    L1Version,
    L1Bio,
    L1Shade,
    L1Cluster,
    L1ChunkTopic,
    L1GenerationResult,
)

logger = logging.getLogger(__name__)

kernel_bp = Blueprint("kernel", __name__, url_prefix="/api/kernel")

# l1_repository = L1Repository()
l1_generator = L1Generator()


def __store_version(
        session, new_version_number: int, description: str = None
) -> L1Version:
    """Store L1 version information"""
    version = L1Version(
        version=new_version_number,
        create_time=datetime.now(),
        status="active",
        description=description or f"L1 data version {new_version_number}",
    )
    session.add(version)
    return version


def __store_bio(session, new_version: int, bio_data: Bio) -> None:
    """Store Bio data"""
    if not bio_data:
        logger.warning("No bio data found")
        return

    bio_record = L1Bio(
        version=new_version,
        content=bio_data.content_second_view,
        content_third_view=bio_data.content_third_view,
        summary=bio_data.summary_second_view,
        summary_third_view=bio_data.summary_third_view,
        create_time=datetime.now(),
    )
    session.add(bio_record)


def __store_shades(session, new_version: int, shades_list: list) -> None:
    """Store Shades data"""
    if not shades_list:
        logger.warning("No shades data found")
        return

    for shade in shades_list:
        shade_data = L1Shade(
            version=new_version,
            name=shade.name,
            aspect=shade.aspect,
            icon=shade.icon,
            desc_third_view=shade.desc_third_view,
            content_third_view=shade.content_third_view,
            desc_second_view=shade.desc_second_view,
            content_second_view=shade.content_second_view,
            create_time=datetime.now(),
        )
        session.add(shade_data)


def __store_clusters(session, new_version: int, cluster_list: list) -> None:
    """Store Clusters data"""
    if not cluster_list:
        logger.warning("No clusters data found")
        return

    for cluster in cluster_list:
        cluster_data = L1Cluster(
            version=new_version,
            cluster_id=cluster.get("clusterId"),
            memory_ids=[m.get("memoryId") for m in cluster.get("memoryList", [])],
            cluster_center=cluster.get("clusterCenter"),
            create_time=datetime.now(),
        )
        session.add(cluster_data)


def __store_chunk_topics(session, new_version: int, chunk_topics_dict: dict) -> None:
    """Store Chunk Topics data"""
    if not isinstance(chunk_topics_dict, dict):
        logger.warning(f"Invalid chunk_topics format: {type(chunk_topics_dict)}")
        return

    logger.info(f"Found chunk topics dict: {chunk_topics_dict}")

    for cluster_id, cluster_data in chunk_topics_dict.items():
        # for each chunkId, create an unique record
        for chunk_id in cluster_data.get("chunkIds", []):
            topic_data = L1ChunkTopic(
                version=new_version,
                chunk_id=chunk_id,
                topic=cluster_data.get("topic"),
                tags=cluster_data.get("tags"),
                create_time=datetime.now(),
            )
            session.add(topic_data)


def store_l1_data(session, l1_data: L1GenerationResult) -> int:
    """Store L1 data based on L0 data"""
    try:
        # 1. Get current latest version
        latest_version = (
            session.query(L1Version).order_by(L1Version.version.desc()).first()
        )

        new_version_number = (latest_version.version + 1) if latest_version else 1
        logger.info(f"Creating new version: {new_version_number}")

        # 2. Create new version record
        version = __store_version(session, new_version_number)

        # 3. Store Bio data
        __store_bio(session, new_version_number, l1_data.bio)

        # 4. Store Shades data
        if hasattr(l1_data.bio, "shades_list"):
            __store_shades(session, new_version_number, l1_data.bio.shades_list)

        # 5. Store Clusters data
        cluster_list = l1_data.clusters.get("clusterList", [])
        __store_clusters(session, new_version_number, cluster_list)

        # 6. Store Chunk Topics data
        __store_chunk_topics(session, new_version_number, l1_data.chunk_topics)

        # 7. Commit transaction
        session.commit()
        logger.info(f"Successfully stored L1 data version {new_version_number}")

        return new_version_number

    except Exception as e:
        session.rollback()
        logger.error(f"Error creating L1 version: {str(e)}")
        raise


def serialize_value(value):
    """iterate over the nested dictionary and serialize the values"""
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),  # turn the shape into a list
            "dtype": str(value.dtype),
        }
    elif isinstance(value, Bio):
        return str(value)
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    elif isinstance(value, (int, float, str, bool)) or value is None:
        return value
    else:
        return str(value)


@kernel_bp.route("/l1/global/generate", methods=["POST"])
def generate_l1():
    """Generate L1 data from L0 data and store"""
    try:
        # 1. Generate L1 data
        result = generate_l1_from_l0()

        if result is None:
            return jsonify(APIResponse.error("No valid L1 data generated"))

        # 2. Store L1 data
        with DatabaseSession.session() as session:
            version_number = store_l1_data(session, result)

        # 3. Convert result to serializable format
        serializable_result = serialize_value(result.to_dict())

        # 4. return the result
        response_data = {
            "version": version_number,
            "message": f"L1 data generated and stored successfully with version {version_number}",
            "data": serializable_result,
        }

        return jsonify(APIResponse.success(data=response_data))

    except Exception as e:
        logger.error(f"Error generating L1: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(str(e)))


@kernel_bp.route("/l1/global/versions", methods=["GET"])
def list_l1_versions():
    """Get all L1 data versions"""
    try:
        with DatabaseSession.session() as session:
            versions = session.query(L1Version).order_by(L1Version.version.desc()).all()

            version_list = [
                {
                    "version": v.version,
                    "create_time": v.create_time.isoformat(),
                    "status": v.status,
                    "description": v.description,
                }
                for v in versions
            ]

            return jsonify(APIResponse.success(data=version_list))

    except Exception as e:
        logger.error(f"Error listing L1 versions: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(str(e)))


@kernel_bp.route("/l1/global/version/<int:version>", methods=["GET"])
def get_l1_version(version):
    """Get specified version of L1 data"""
    try:
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

            return jsonify(APIResponse.success(data=data))

    except Exception as e:
        logger.error(f"Error getting L1 version {version}: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(str(e)))


@kernel_bp.route("/l1/status_bio/generate", methods=["POST"])
def generate_status_biography():
    """Generate status biography"""
    # Call l1_manager method to generate and store status biography
    status_bio = generate_and_store_status_bio()

    # Build response data
    response_data = {
        "content": status_bio.content_second_view,
        "content_third_view": status_bio.content_third_view,
        "summary": status_bio.summary_second_view,
        "summary_third_view": status_bio.summary_third_view,
        "shades": [
            {
                "name": shade.name,
                "aspect": shade.aspect,
                "icon": shade.icon,
                "desc_third_view": shade.desc_third_view,
                "content_third_view": shade.content_third_view,
                "desc_second_view": shade.desc_second_view,
                "content_second_view": shade.content_second_view,
            }
            for shade in status_bio.shades_list
        ],
    }

    return jsonify(APIResponse.success(data=response_data))


@kernel_bp.route("/l1/status_bio/get", methods=["GET"])
def get_status_biography():
    """Get status biography"""
    # Call l1_manager method to get status biography
    status_bio = get_latest_status_bio()
    return jsonify(APIResponse.success(data=status_bio))


@kernel_bp.route("/l1/latest/save_topics", methods=["GET"])
def save_latest_topics():
    """get the latest L1 topics and save to the pointed directory"""
    # use the related resources directory
    base_dir = os.path.join(os.getcwd(), "resources/L2/data_pipeline/raw_data")
    os.makedirs(base_dir, exist_ok=True)

    chunk_service = ChunkService()
    topics_data = chunk_service.query_topics_data()
    save_true_topics(topics_data, os.path.join(base_dir, "topics.json"))

    return jsonify(APIResponse.success(data={}))


@kernel_bp.route("/l1/latest/save_notes", methods=["GET"])
def save_latest_notes():
    """Get latest version of notes and save to file"""
    documents = document_service.list_documents_with_l0()
    notes_list, _ = extract_notes_from_documents(documents)
    if not notes_list:
        return jsonify(APIResponse.error("No notes found"))
    # Get latest topics information
    note_service = NoteService()
    note_service.prepareNotes(notes_list)

    # use NotesStorage to save notes
    storage = NotesStorage()
    result = storage.save_notes(notes_list)

    return jsonify(APIResponse.success(data=result))


@kernel_bp.route("/l1/notes", methods=["GET"])
def get_notes():
    """Get notes from file"""
    storage = NotesStorage()
    try:
        notes_list = storage.load_notes()
    except FileNotFoundError:
        return jsonify(
            APIResponse.error("Notes file not found. Please save notes first.")
        )

    # serialize notes
    serializable_notes = [NoteSerializer.to_dict(note) for note in notes_list]

    return jsonify(
        APIResponse.success(
            data={"notes": serializable_notes, "count": len(serializable_notes)}
        )
    )
