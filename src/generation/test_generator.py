import logging
from src.generation.generator import generate_answers

# Setup test logging
logging.basicConfig(level=logging.INFO)

def test_generate_answers():
    query = "What's happening with Apple stock today?"
    contexts = [
        "Apple Inc. stocks surged 5% in the last quarter.",
        "Tesla has a strong outlook for 2025 despite global challenges.",
        "Amazon is seeing record-breaking sales due to new product launches."
    ]
    
    answers = generate_answers(contexts, query, top_k=2)
    print(f"Query: {query}")
    for i, ans in enumerate(answers, 1):
        print(f"Answer {i}: {ans}")

if __name__ == "__main__":
    test_generate_answers()
