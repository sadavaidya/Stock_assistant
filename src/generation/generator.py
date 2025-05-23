import logging
from transformers import pipeline, AutoTokenizer

logger = logging.getLogger(__name__)

# Load a more powerful generative model
model_name = "google/flan-t5-large"
qa_pipeline = pipeline("text2text-generation", model=model_name, tokenizer=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

MAX_INPUT_LENGTH = 512  # Limit for model input
CHUNK_SIZE = 400        # Ensures chunks stay within token limit


def generate_answer(query, documents):
    try:
        if not documents:
            logger.warning("No documents retrieved for generation.")
            return "Sorry, I couldn't find relevant information."

        # Combine documents into context
        context = ""
        for doc in documents:
            context += f"[{doc['date']} | {doc['source']}] {doc['text']}\n"

        # Tokenize context to check total length
        tokenized_context = tokenizer(context, return_tensors="pt", truncation=True, padding=True)
        context_length = len(tokenized_context['input_ids'][0])

        # Chunk if needed
        if context_length > MAX_INPUT_LENGTH:
            logger.warning("Context is too long. Splitting into smaller chunks.")
            chunks = [context[i:i + CHUNK_SIZE] for i in range(0, len(context), CHUNK_SIZE)]
        else:
            chunks = [context]

        answers = []

        for chunk in chunks:
            prompt = (
                f"You are an expert financial assistant.\n\n"
                f"Below are news documents strictly related to the user's query.\n"
                f"Rules:\n"
                f"- Use ONLY the given information.\n"
                f"- DO NOT invent facts.\n"
                f"- If the answer is not clearly found, reply exactly: 'Information not available in provided documents.'\n\n"
                f"Documents:\n{chunk}\n\n"
                f"Question: {query}\n\n"
                f"Provide a short, clear, fact-based answer:"
            )

            logger.debug(f"Prompt to model:\n{prompt}")

            result = qa_pipeline(
                prompt,
                max_length=256,
                do_sample=False,
                num_beams=4,
                early_stopping=True
            )

            answer = result[0]["generated_text"]
            answers.append(answer)

        combined_answer = " ".join(answers).strip()
        logger.info(f"Generated answer: {combined_answer}")
        return combined_answer

    except Exception as e:
        logger.error(f"Error during QA generation: {e}")
        return "Error generating answer"
