# src/embeddings/embed_text.py

import os
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "src/embeddings/embeddings.npy"
DOCS_PATH = "src/vector_store/documents.pkl"

def embed_documents(docs, model_name=MODEL_NAME):
    logger.info(f"Embedding {len(docs)} texts...")
    model = SentenceTransformer(model_name)

    texts = [doc["text"] for doc in docs]
    embeddings = model.encode(texts, show_progress_bar=True)

    logger.info("Embeddings generation complete.")
    return embeddings, docs

def save_embeddings(embeddings, documents, embeddings_path=EMBEDDINGS_PATH, docs_path=DOCS_PATH):
    np.save(embeddings_path, embeddings)
    with open(docs_path, "wb") as f:
        pickle.dump(documents, f)

    logger.info(f"Embeddings saved to {embeddings_path}")
    logger.info(f"Documents saved to {docs_path}")

# if __name__ == "__main__":
    # Example test run with fake data
    # sample_docs = [
    #     {
    #         "text": "Apple stock rose 3% today.",
    #         "source": "yfinance",
    #         "date": "2025-04-23",
    #         "ticker": "AAPL"
    #     },
    #     {
    #         "text": "Experts say Apple is on track for a strong Q2.",
    #         "source": "marketaux",
    #         "date": "2025-04-23",
    #         "ticker": "AAPL"
    #     }
    # ]
    # embeddings, updated_docs = embed_documents(sample_docs)
    # save_embeddings(embeddings, updated_docs)
