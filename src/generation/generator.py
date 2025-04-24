from transformers import pipeline, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

# Load generative model
model_name = "google/flan-t5-base"
qa_pipeline = pipeline("text2text-generation", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

MAX_INPUT_LENGTH = 512  # Adjust this based on model's max length
CHUNK_SIZE = 400  # We'll use a chunk size that ensures we stay within limits

def generate_answer(query, documents):
    try:
        if not documents:
            logger.warning("No documents retrieved for generation.")
            return "Sorry, I couldn't find relevant information."

        # Build context and prepare prompt
        context = ""
        for doc in documents:
            context += f"[{doc['date']} | {doc['source']}] {doc['text']}\n"

        # Tokenize context to check length
        tokenized_context = tokenizer(context, return_tensors="pt", truncation=True, padding=True)
        context_length = len(tokenized_context['input_ids'][0])

        # If the context is too long, split it into chunks
        if context_length > MAX_INPUT_LENGTH:
            logger.warning("Context is too long. Splitting into smaller chunks.")
            chunks = [context[i:i + CHUNK_SIZE] for i in range(0, len(context), CHUNK_SIZE)]
        else:
            chunks = [context]

        # Generate answers for each chunk
        answers = []
        for chunk in chunks:
            prompt = (
            "You are a financial advisor. Based on this context:{chunk} advise the user on his question: {query}"
            )

            logger.debug(f"Prompt to model:\n{prompt}")
            result = qa_pipeline(prompt, max_length=256, do_sample=False)
            answer = result[0]["generated_text"]
            answers.append(answer)
        
        # Combine all the answers from different chunks
        combined_answer = " ".join(answers)
        logger.info(f"Generated answer: {combined_answer}")
        return combined_answer

    except Exception as e:
        logger.error(f"Error during QA generation: {e}")
        return "Error generating answer"
