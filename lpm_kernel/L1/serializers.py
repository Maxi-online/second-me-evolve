from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import os

import numpy as np

from .bio import Chunk, Note


@dataclass
class ChunkSerializer:
    """Chunk serialization/deserialization class."""

    @staticmethod
    def to_dict(chunk: Chunk) -> Dict[str, Any]:
        """Serialize Chunk object to dictionary.
        
        Args:
            chunk: The Chunk object to serialize.
            
        Returns:
            Dictionary representation of the Chunk.
        """
        return {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "embedding": chunk.embedding.tolist()
            if chunk.embedding is not None
            else None,
            "tags": chunk.tags,
            "topic": chunk.topic,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Chunk:
        """Deserialize dictionary to Chunk object.
        
        Args:
            data: Dictionary containing chunk data.
            
        Returns:
            Reconstructed Chunk object.
        """
        return Chunk(
            id=data["id"],
            document_id=data["document_id"],
            content=data["content"],
            embedding=np.array(data["embedding"]) if data.get("embedding") else None,
            tags=data.get("tags"),
            topic=data.get("topic"),
        )


@dataclass
class NoteSerializer:
    """Note serialization/deserialization class."""

    @staticmethod
    def to_dict(note: Note) -> Dict[str, Any]:
        """Serialize Note object to dictionary.
        
        Args:
            note: The Note object to serialize.
            
        Returns:
            Dictionary representation of the Note.
        """
        return {
            "noteId": note.id,
            "content": note.content,
            "createTime": note.create_time,
            "memoryType": note.memory_type,
            "embedding": note.embedding.tolist()
            if note.embedding is not None
            else None,
            "title": note.title,
            "summary": note.summary,
            "insight": note.insight,
            "tags": note.tags if note.tags else [],
            "topic": note.topic,
            "chunks": [ChunkSerializer.to_dict(chunk) for chunk in note.chunks],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Note:
        """Deserialize dictionary to Note object.
        
        Args:
            data: Dictionary containing note data.
            
        Returns:
            Reconstructed Note object.
        """
        chunks = [
            ChunkSerializer.from_dict(chunk_data)
            for chunk_data in data.get("chunks", [])
        ]

        return Note(
            noteId=data["noteId"],
            content=data["content"],
            createTime=data["createTime"],
            memoryType=data["memoryType"],
            embedding=np.array(data["embedding"]) if data.get("embedding") else None,
            chunks=chunks,
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            insight=data.get("insight", ""),
            tags=data.get("tags", []),
            topic=data.get("topic"),
        )


class NotesStorage:
    """Notes storage management class."""

    def __init__(self, base_dir: str = None):
        """Initialize the NotesStorage.
        
        Args:
            base_dir: Base directory for storing notes. If None, uses a default path.
        """
        if base_dir is None:
            base_dir = os.path.join(os.getcwd(), "resources/L2/data_pipeline/raw_data")
        self.base_dir = base_dir
        self.notes_path = os.path.join(base_dir, "notes.json")
        self.topics_path = os.path.join(base_dir, "topics.json")

    def save_notes(self, notes: List[Note]) -> Dict[str, Any]:
        """Save Notes list to file.
        
        Args:
            notes: List of Note objects to save.
            
        Returns:
            Dictionary containing save status, count, and validation results.
        """
        # Ensure directory exists
        os.makedirs(self.base_dir, exist_ok=True)

        # Collect validation information
        validation_info = {
            "total_notes": len(notes),
            "total_chunks": 0,
            "note_ids": set(),
            "chunk_ids": set(),
        }

        # Serialize notes
        serializable_notes = []
        for note in notes:
            validation_info["note_ids"].add(str(note.id))
            validation_info["total_chunks"] += len(note.chunks)
            for chunk in note.chunks:
                validation_info["chunk_ids"].add(str(chunk.id))

            serializable_notes.append(NoteSerializer.to_dict(note))

        # Save to file
        with open(self.notes_path, "w", encoding="utf-8") as f:
            json.dump(serializable_notes, f, ensure_ascii=False, indent=2)

        # Validate saved data
        with open(self.notes_path, "r", encoding="utf-8") as f:
            saved_notes = json.load(f)

        saved_validation = {
            "total_notes": len(saved_notes),
            "total_chunks": sum(len(note["chunks"]) for note in saved_notes),
            "note_ids": {str(note["noteId"]) for note in saved_notes},
            "chunk_ids": {
                str(chunk["id"]) for note in saved_notes for chunk in note["chunks"]
            },
        }

        validation_result = {
            "notes_count_match": validation_info["total_notes"]
            == saved_validation["total_notes"],
            "chunks_count_match": validation_info["total_chunks"]
            == saved_validation["total_chunks"],
            "note_ids_match": validation_info["note_ids"]
            == saved_validation["note_ids"],
            "chunk_ids_match": validation_info["chunk_ids"]
            == saved_validation["chunk_ids"],
        }

        return {
            "message": f"Notes saved to {self.notes_path}",
            "count": len(serializable_notes),
            "validation": validation_result,
            "stats": {
                "original": {
                    k: len(v) if isinstance(v, set) else v
                    for k, v in validation_info.items()
                },
                "saved": {
                    k: len(v) if isinstance(v, set) else v
                    for k, v in saved_validation.items()
                },
            },
        }

    def load_notes(self) -> List[Note]:
        """Load Notes list from file.
        
        Returns:
            List of Note objects loaded from file.
            
        Raises:
            FileNotFoundError: If the notes file doesn't exist.
        """
        if not os.path.exists(self.notes_path):
            raise FileNotFoundError(f"Notes file not found at {self.notes_path}")

        with open(self.notes_path, "r", encoding="utf-8") as f:
            notes_data = json.load(f)

        return [NoteSerializer.from_dict(note_data) for note_data in notes_data]
