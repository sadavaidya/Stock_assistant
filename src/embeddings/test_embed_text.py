# src/embeddings/test_embed_text.py

from embed_text import embed_text, save_embeddings, load_embeddings

if __name__ == "__main__":
    # Example data
    texts = [
        "Apple Inc. stocks surged 5% in the last quarter.",
        "Tesla has a strong outlook for 2025 despite global challenges."
    ]

    print("\n=== Embedding Texts ===")
    embeddings = embed_text(texts)
    print(embeddings[:2])  # Print first two embeddings for check

    # Save embeddings to file
    save_embeddings(embeddings, "src/embeddings/embeddings.npy")

    # Load the embeddings back
    loaded_embeddings = load_embeddings("src/embeddings/embeddings.npy")
    print("\n=== Loaded Embeddings ===")
    print(loaded_embeddings[:2])  # Print first two embeddings
