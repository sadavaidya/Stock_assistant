import faiss
import numpy as np
import pickle
import logging
from sentence_transformers import SentenceTransformer
from src.embeddings.embed_text import embed_documents
from datetime import datetime

logger = logging.getLogger(__name__)

# Paths
INDEX_PATH = "src/vector_store/faiss_index.index"
DOCS_PATH = "src/vector_store/documents.pkl"

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_documents(query, top_k=3, ticker=None):
    """
    Retrieve top-k documents based on query and optionally rerank based on ticker mention.
    """
    # Load FAISS index
    index = faiss.read_index(INDEX_PATH)
    logger.info("FAISS index loaded successfully.")

    # Load documents
    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)
    logger.info(f"Loaded {len(documents)} documents.")

    # Embed query
    query_embedding, _ = embed_documents(
        [{"text": query, "source": "user", "date": str(datetime.now().date()), "ticker": "unknown"}],
        "all-MiniLM-L6-v2"
    )

    # Search in FAISS
    distances, indices = index.search(query_embedding, top_k)
    logger.info(f"Top {top_k} documents retrieved for query: {query}")
    logger.info(f"Distances: {distances}")
    logger.info(f"Indices: {indices}")

    # Collect matched documents (handling out of range safely)
    results = []
    for idx in indices[0]:
        if idx < len(documents):
            results.append(documents[idx])
        else:
            logger.warning(f"Index {idx} out of bounds for loaded documents.")

    # If ticker provided, rerank based on presence in text
    if ticker:
        results = rerank_by_ticker(results, ticker)

    return results


def rerank_by_ticker(docs, ticker):
    """
    Rerank documents, prioritizing those mentioning the ticker.
    """
    reranked = sorted(docs, key=lambda doc: ticker.lower() in doc.get("text", "").lower(), reverse=True)
    logger.info(f"Documents reranked based on presence of ticker '{ticker}'.")
    return reranked
