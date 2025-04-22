# src/vector_store/test_faiss_store.py

from faiss_store import FAISSVectorStore
from src.embeddings.embed_text import embed_text

if __name__ == "__main__":
    # Example texts (headlines, company info)
    texts = [
        "Apple Inc. stocks surged 5% in the last quarter.",
        "Tesla has a strong outlook for 2025 despite global challenges."
    ]
    
    # Embed texts
    embeddings = embed_text(texts)
    
    # Initialize FAISS vector store
    faiss_store = FAISSVectorStore(embedding_dimension=384)
    
    # Add embeddings to the FAISS index
    faiss_store.add_embeddings(embeddings)
    
    # Save the index
    faiss_store.save_index()

    # Load the index
    faiss_store.load_index()

    # Search for a query in the FAISS index
    query = "How did Apple's stock perform recently?"
    distances, indices = faiss_store.search(query, top_k=2)
    
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")
