import logging
from src.data.fetch_news import fetch_news_from_marketaux  # For fetching news data
from src.data.fetch_stocks import fetch_company_info, fetch_historical_prices  # For fetching stock data
from src.embeddings.embed_text import embed_text, save_documents, save_embeddings  # For embedding documents
from src.vector_store.faiss_store import FAISSVectorStore # For FAISS index
from src.retrieval.retriever import retrieve_similar_documents  # For retrieving docs
from src.generation.generator import generate_answer  # For generating the response

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_rag_pipeline(query: str, top_k: int = 3) -> str:
    """
    Full RAG pipeline for retrieving relevant documents and generating answers.

    Args:
        query (str): The user query to generate a response for.
        top_k (int): The number of top documents to retrieve.

    Returns:
        str: The generated answer.
    """
    # Step 1: Fetch news and stock data
    logger.info("Fetching data...")
    news_data = fetch_news_from_marketaux("AAPL")  # Fetch news for Apple stock
    stock_data = fetch_company_info("AAPL")  # Fetch stock data for Apple

    # Step 2: Combine the fetched data (you can customize this step based on your needs)
    documents = news_data 
    logger.info(f"Fetched {len(documents)} documents.")

    # Step 3: Embed the documents
    logger.info("Embedding documents...")
    embeddings = embed_text(documents)
    save_embeddings(embeddings, "src/embeddings/embeddings.npy")
    save_documents(documents, "src/vector_store/documents.pkl")

    # Step 4: Create or load FAISS index
    faiss_store = FAISSVectorStore(embedding_dimension=384)
    # Add embeddings to the FAISS index
    faiss_store.add_embeddings(embeddings)
    # Save the index
    faiss_store.save_index()
    # Load the index
    faiss_store.load_index()
    
    # Step 5: Store the documents in FAISS

    # Step 6: Retrieve relevant documents based on the query
    logger.info("Retrieving relevant documents...")
    retrieved_docs = retrieve_similar_documents(query, top_k=top_k)

    # Step 7: Generate an answer using the retrieved documents
    logger.info("Generating answer...")
    answer = generate_answer(query, retrieved_docs)
    
    return answer

if __name__ == "__main__":
    # Test the pipeline with a sample query
    query = "What is the current status of Apple stock?"
    generated_answer = run_rag_pipeline(query)
    print(f"Query: {query}")
    print(f"Generated Answer: {generated_answer}")
