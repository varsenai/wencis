"""
Vector Store — ARIA's long-term semantic memory.

Uses ChromaDB for local vector storage and retrieval. 
This allows ARIA to perform semantic searches across the entire workspace,
providing "Infinite Recall" beyond the active context window.
"""

import os
import uuid
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None

from silex.utils.config import VYN_VECTOR_DB
from silex.utils.logger import setup_logger

log = setup_logger("silex.memory.vector")

class VectorStore:
    """
    Manages local vector storage for semantic retrieval of workspace content.
    """

    def __init__(self, collection_name: str = "aria_workspace"):
        if chromadb is None:
            log.error("ChromaDB not installed. VectorStore will be inactive.")
            self.client = None
            return

        self.persist_path = str(VYN_VECTOR_DB)
        os.makedirs(self.persist_path, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_path)
        
        # Use a local embedding function (MiniLM) to stay sovereign
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        log.info(f"VectorStore initialized at {self.persist_path} (Collection: {collection_name})")

    @property
    def is_active(self) -> bool:
        return self.client is not None

    def add_chunks(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """Adds a list of text chunks to the vector store."""
        if not self.client:
            return
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
            
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        log.debug(f"Added {len(texts)} chunks to VectorStore.")

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Performs a semantic search for the given query."""
        if not self.client:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        formatted_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i] if 'ids' in results and results['ids'] else None,
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted_results

    def delete_by_path(self, file_path: str):
        """Removes all chunks associated with a specific file path."""
        if not self.client:
            return
        self.collection.delete(where={"path": file_path})
        log.debug(f"Deleted vector entries for path: {file_path}")

    def delete_by_ids(self, ids: List[str]):
        """Removes chunks by their exact IDs."""
        if not self.client:
            return
        self.collection.delete(ids=ids)
        log.debug(f"Deleted vector entries for ids: {ids}")

    def clear(self):
        """Wipes the entire collection."""
        if not self.client:
            return
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_function
        )
        log.warning("VectorStore collection cleared.")
