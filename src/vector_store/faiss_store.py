# src/vector_store/faiss_store.py

import os
import faiss
import numpy as np
from typing import List, Optional
import logging

from src.embeddings.embed_text import embed_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, embedding_dimension: int, index_file: str = "src/vector_store/faiss_index.index"):
        self.embedding_dimension = embedding_dimension
        self.index_file = index_file
        self.index = None

    def create_index(self) -> None:
        """Create a FAISS index."""
        try:
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
            logger.info("FAISS index created.")
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")

    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """Add embeddings to the FAISS index."""
        try:
            if self.index is None:
                self.create_index()

            self.index.add(embeddings)
            logger.info(f"Added {len(embeddings)} embeddings to the FAISS index.")
        except Exception as e:
            logger.error(f"Error adding embeddings to FAISS index: {e}")

    def save_index(self) -> None:
        """Save the FAISS index to disk."""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_file)
                logger.info(f"FAISS index saved to {self.index_file}")
            else:
                logger.error("Index is None. Cannot save.")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")

    def load_index(self) -> None:
        """Load the FAISS index from disk."""
        try:
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                logger.info(f"FAISS index loaded from {self.index_file}")
            else:
                logger.error(f"Index file {self.index_file} does not exist.")
        except Exception as e:
            logger.error(f"Error loading FAISS index from {self.index_file}: {e}")

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search the FAISS index for the most relevant results based on the query."""
        try:
            if self.index is None:
                logger.error("FAISS index is not loaded.")
                return []

            query_embedding = embed_text([query])
            distances, indices = self.index.search(query_embedding, top_k)
            logger.info(f"Top {top_k} results retrieved from FAISS index.")
            
            return distances, indices
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}")
            return [], []
