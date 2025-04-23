# import logging
# from transformers import pipeline
# from typing import List

# # Set up logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# # Initialize the question-answering pipeline using a Hugging Face model
# try:
#     qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
#     logger.info("Loaded QA model: deepset/roberta-base-squad2")
# except Exception as e:
#     logger.error(f"Error loading QA model: {e}")
#     raise

# def generate_answers(contexts: List[str], query: str, top_k: int = 3) -> List[str]:
#     """
#     Generate answers for a query based on given document contexts.

#     Args:
#         contexts (List[str]): List of context strings.
#         query (str): User question.
#         top_k (int): Number of answers to generate.

#     Returns:
#         List[str]: Answers generated from top-k contexts.
#     """
#     answers = []
#     logger.info(f"Generating answers for query: {query}")

#     try:
#         for i, context in enumerate(contexts[:top_k]):
#             result = qa_pipeline(question=query, context=context)
#             answer = result.get("answer", "")
#             score = result.get("score", 0.0)
#             logger.info(f"Context {i+1}: Answer: '{answer}' (Score: {score:.4f})")
#             answers.append(answer)
#     except Exception as e:
#         logger.error(f"Error during QA generation: {e}")

#     return answers


from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

# Initialize the QA pipeline (Roberta or any other model as per your choice)
generator = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")

def generate_answer(query: str, documents: list) -> str:
    """
    Generate an answer based on the retrieved documents and query.

    Args:
        query (str): The user query.
        documents (list): The list of retrieved documents.

    Returns:
        str: The generated answer.
    """
    try:
        # Combine retrieved documents into a single context
        context = " ".join([doc["content"] for doc in documents])

        # Prepare the input for generation
        input_text = {
            "question": query,
            "context": context
        }

        # Generate the answer using the QA model
        result = generator(input_text)

        # Extract and return the answer
        answer = result['answer']
        logger.info(f"Generated answer: {answer}")
        return answer

    except Exception as e:
        logger.error(f"Error during QA generation: {e}")
        return "Error generating answer"
