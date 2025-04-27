# src/rag/active_rag.py

import os
import logging
from pathlib import Path
import pickle

from src.data.fetch_news import fetch_news_from_marketaux
from src.data.fetch_stocks import fetch_stock_summary_from_yfinance
from src.embeddings.embed_text import embed_documents, save_embeddings
from src.vector_store.faiss_store import FAISSVectorStore
from src.retrieval.retriever import retrieve_documents
from src.generation.generator import generate_answer
from src.utils.extract_ticker import get_ticker_from_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = "src/embeddings/embeddings.npy"
DOCS_PATH = Path("src/vector_store/documents.pkl")

def save_documents(documents):
    """Helper to save documents to disk."""
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

def load_documents():
    """Helper to load documents from disk."""
    if not DOCS_PATH.exists():
        return []
    with open(DOCS_PATH, "rb") as f:
        return pickle.load(f)

def active_rag(query: str, top_k: int = 3, max_iterations: int = 2):

    ticker = get_ticker_from_query(query)
    if not ticker:
        logger.warning("Could not extract ticker from query.")
        return

    logger.info(f"Detected ticker: {ticker}")

    # Step 1: Fetch documents
    logger.info("Fetching documents...")
    news_docs = fetch_news_from_marketaux(ticker)
    stock_docs = fetch_stock_summary_from_yfinance(ticker)
    all_docs = news_docs + stock_docs

    if not all_docs:
        logger.warning("No documents fetched. Exiting.")
        return

    # Step 2: Embed and save
    logger.info("Embedding documents...")
    embeddings, docs = embed_documents(all_docs)
    save_embeddings(documents=docs, embeddings=embeddings)

    # Step 3: Build and save FAISS index
    logger.info("Building FAISS index...")
    faiss_store = FAISSVectorStore(embedding_dimension=384)
    faiss_store.add_embeddings(embeddings)
    faiss_store.save_index()

    # Step 4: Save the documents for retriever
    logger.info("Saving documents...")
    save_documents(docs)

    # Step 5: Active Retrieval & Refinement Loop
    for iteration in range(max_iterations):
        logger.info(f"=== Active RAG Iteration {iteration + 1}/{max_iterations} ===")

        # Load FAISS index and documents
        faiss_store.load_index()
        all_docs = load_documents()

        # Retrieve
        logger.info("Retrieving relevant documents...")
        retrieved_docs = retrieve_documents(query, top_k=top_k)

        # Check coverage: enough retrieved documents?
        if len(retrieved_docs) >= top_k:
            logger.info("Sufficient relevant documents retrieved.")
            break  # stop active refinement

        # If not enough, fetch more news/stocks and refine
        logger.info("Fetching additional documents...")
        additional_news = fetch_news_from_marketaux(ticker)
        additional_stocks = fetch_stock_summary_from_yfinance(ticker)
        new_docs = additional_news + additional_stocks

        if not new_docs:
            logger.warning("No additional documents found.")
            break

        # Embed new docs
        logger.info("Embedding additional documents...")
        new_embeddings, new_docs = embed_documents(new_docs)

        # Add to FAISS
        faiss_store.add_embeddings(new_embeddings)
        faiss_store.save_index()

        # Add to documents.pkl
        all_docs.extend(new_docs)
        save_documents(all_docs)

    # Final retrieval after loop
    retrieved_docs = retrieve_documents(query, top_k=top_k)

    # Step 6: Generate final answer
    logger.info("Generating answer...")
    answer = generate_answer(query, retrieved_docs)

    # Output
    print(f"\n📌 Query: {query}")
    print(f"✅ Answer: {answer}")

    print("\n📄 Relevant Documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n🔹 Document {i + 1}")
        print(f"📅 Date: {doc.get('date')}")
        print(f"📰 Source: {doc.get('source')}")
        print(f"💬 Text: {doc.get('text')}")


if __name__ == "__main__":
    user_query = input("Enter your question: ")
    active_rag(query=user_query)
