import logging
from src.generation.generator import generate_answer

# Setup test logging
logging.basicConfig(level=logging.INFO)

def test_generate_answers():
    from src.generation.generator import generate_answer
    import logging
    logging.basicConfig(level=logging.INFO)

    query = "Should I invest in Tesla stocks now?"
    contexts = [
        "Tesla stock has been fluctuating but maintains a strong upward trend.",
        "Apple recently launched a new chip that boosted its stock prices.",
        "NVIDIA shares are being driven by AI demand."
    ]

    answer = generate_answer(query, contexts)
    print(f"📌 Query: {query}\n✅ Answer: {answer}")


if __name__ == "__main__":
    test_generate_answers()
