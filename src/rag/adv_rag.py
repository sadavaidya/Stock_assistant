# adv_rag.py


from src.data.fetch_news import fetch_news_from_marketaux
from src.data.fetch_stocks import fetch_stock_summary_from_yfinance
from src.embeddings.embed_text import embed_documents, save_embeddings
from src.vector_store.faiss_store import FAISSVectorStore
from src.retrieval.retriever import retrieve_documents
from src.generation.generator import generate_answer
from src.utils.extract_ticker import get_tickers_from_query

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


EMBEDDINGS_PATH = "src/embeddings/embeddings.npy"
DOCS_PATH = "src/vector_store/documents.pkl"


def run_advanced_rag(query: str, top_k: int = 3):

    ticker = get_tickers_from_query(query)
    if not ticker:
        logger.warning("Could not extract ticker from query.")
        return

    logger.info(f"Detected ticker: {ticker}")

    # Step 1: Fetch documents
    logger.info("Fetching documents...")
    for ticker in ticker:
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

    # Step 3: Build or load FAISS index
    logger.info("Building FAISS index...")
    faiss_store = FAISSVectorStore(embedding_dimension=384)
    
    # Add embeddings to the FAISS index
    faiss_store.add_embeddings(embeddings)
    
    # Save the index
    faiss_store.save_index()

    # Load the index
    faiss_store.load_index()

    # Step 4: Retrieve top-k relevant docs for query
    logger.info("Retrieving relevant documents...")
    retrieved_docs = retrieve_documents(query, top_k=top_k)

    # Step 5: Generate answer
    logger.info("Generating answer...")
    answer = generate_answer(query, retrieved_docs)
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
    run_advanced_rag(query=user_query)

