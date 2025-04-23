# src/embeddings/embed_text.py

import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the embedding model (using a good pre-trained model like 'all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts using the Sentence Transformers model.

    Args:
        texts (List[str]): List of strings (e.g., headlines, company info)

    Returns:
        np.ndarray: Embeddings corresponding to the input texts
    """
    try:
        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = model.encode(texts, convert_to_numpy=True)
        logger.info("Embeddings generation complete.")
        return embeddings
    except Exception as e:
        logger.error(f"Error embedding texts: {e}")
        return np.array([])

def save_documents(documents: list, filename: str) -> None:
    """
    Save the list of documents (text chunks) to a file.

    Args:
        documents (list): List of text chunks corresponding to embeddings
        filename (str): Path to save the document list (e.g., documents.pkl)
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(documents, f)
        logger.info(f"Documents saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving documents: {e}")


def save_embeddings(embeddings: np.ndarray, filename: str) -> None:
    """
    Save the embeddings to a file.

    Args:
        embeddings (np.ndarray): Embedding vectors
        filename (str): Path to the file (e.g., embeddings.npy)
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.save(filename, embeddings)
        logger.info(f"Embeddings saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")

def load_embeddings(filename: str) -> np.ndarray:
    """
    Load pre-saved embeddings from a file.

    Args:
        filename (str): Path to the saved embeddings file

    Returns:
        np.ndarray: Loaded embeddings
    """
    try:
        embeddings = np.load(filename)
        logger.info(f"Embeddings loaded from {filename}")
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings from {filename}: {e}")
        return np.array([])

