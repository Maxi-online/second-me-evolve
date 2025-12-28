import chromadb
from chromadb.config import Settings
from chromadb.errors import IDAlreadyExistsError
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VectorDocument:
    id: str
    text: str
    metadata: Dict
    embedding: Optional[List[float]] = None


class BaseVectorRepository(ABC):
    @abstractmethod
    def add(self, documents: List[VectorDocument]) -> None:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], limit: int = 5) -> List[VectorDocument]:
        pass


class ChromaRepository(BaseVectorRepository):
    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Check if collection exists, create it if it doesn't
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except ValueError:  # ValueError is thrown when Collection does not exist
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "dimension": 1536},
            )

    def add(self, documents: List[VectorDocument]) -> None:
        """
        Add documents to the vector store
        """
        if not documents:
            return

        ids = [doc.id for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = [doc.embedding for doc in documents if doc.embedding is not None]

        # If embeddings are provided, use them directly
        if embeddings and len(embeddings) == len(documents):
            self.collection.add(
                ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
            )
        else:
            # Let ChromaDB handle embedding generation
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def search(self, query_vector: List[float], limit: int = 5) -> List[VectorDocument]:
        """
        Search similar documents using a query vector
        """
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        for i in range(len(results["ids"][0])):
            doc = VectorDocument(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                embedding=None,  # ChromaDB doesn't return embeddings in search results
            )
            documents.append(doc)

        return documents

    def get_by_ids(self, ids: List[str]) -> List[VectorDocument]:
        """
        Retrieve documents by their IDs
        """
        results = self.collection.get(ids=ids, include=["documents", "metadatas"])

        documents = []
        for i in range(len(results["ids"])):
            doc = VectorDocument(
                id=results["ids"][i],
                text=results["documents"][i],
                metadata=results["metadatas"][i],
                embedding=None,
            )
            documents.append(doc)

        return documents

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by their IDs
        """
        self.collection.delete(ids=ids)
