# src/llm_handler.py

import ollama


def get_answer(query, relevant_chunks):
    """Send question + relevant chunks to Phi and get answer"""

    # combine all relevant chunks into one context
    context = "\n\n".join(relevant_chunks)

    # build the prompt
    prompt = f"""You are a helpful assistant that answers questions 
based on the provided document context only.

Context from document:
{context}

Question: {query}

Answer clearly and concisely based only on the context above.
If the answer is not in the context, say 'I could not find 
this information in the document.'
"""

    print("Sending to Phi model...")

    try:
        response = ollama.chat(
            model="phi",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        answer = response["message"]["content"]
        return answer
    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg or "10061" in error_msg:
            raise Exception(
                "❌ Ollama is not running!\n\n"
                "To fix this:\n"
                "1. Download Ollama from https://ollama.ai\n"
                "2. Start the Ollama application\n"
                "3. Run: ollama pull phi\n"
                "4. Try your question again"
            ) from e
        else:
            raise Exception(f"Error communicating with Ollama: {str(e)}") from e