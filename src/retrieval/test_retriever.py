# test_retriever.py

from src.retrieval.retriever import retrieve_similar_documents

def test_retrieve_documents():
    query = "What's happening with Apple stock today?"
    top_k = 3

    results = retrieve_similar_documents(query, top_k=top_k)

    print(f"\nQuery: {query}")
    print(f"Top {top_k} Retrieved Documents:\n")

    for i, doc in enumerate(results):
        print(f"{i + 1}. {doc}\n")

if __name__ == "__main__":
    test_retrieve_documents()
