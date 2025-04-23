import logging
from pathlib import Path
from typing import List

import faiss
import pickle
from sentence_transformers import SentenceTransformer
from src.embeddings.embed_text import embed_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retriever")

# Paths
INDEX_PATH = Path("src/vector_store/faiss_index.index")
DOCS_PATH = Path("src/vector_store/documents.pkl")

# Load FAISS index
def load_faiss_index():
    try:
        index = faiss.read_index(str(INDEX_PATH))
        logger.info("FAISS index loaded successfully.")
        return index
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        raise

# Load corresponding documents
def load_documents() -> List[str]:
    try:
        with open(DOCS_PATH, "rb") as f:
            documents = pickle.load(f)
        logger.info(f"Loaded {len(documents)} documents.")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise

# Query FAISS index
def retrieve_similar_documents(query: str, top_k: int = 3) -> List[str]:
    # Embed the query
    query_embedding = embed_text([query])
    
    # Load FAISS index and documents
    index = load_faiss_index()
    documents = load_documents()

    # Search
    distances, indices = index.search(query_embedding, top_k)
    logger.info(f"Top {top_k} documents retrieved for query: {query}")
    logger.info(f"Distances: {distances}")
    logger.info(f"Indices: {indices}")

    # Get results
    results = [documents[idx] for idx in indices[0] if idx < len(documents)]
    return results
