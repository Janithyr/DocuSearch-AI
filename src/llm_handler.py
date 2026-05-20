# src/llm_handler.py

import os

from groq import Groq


def get_answer(query, relevant_chunks):
    """Send question + relevant chunks to Groq and get answer"""

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

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        raise Exception(
            "❌ GROQ_API_KEY is missing.\n\n"
            "Add it to your environment (local) or Streamlit Secrets (cloud)."
        )

    print("Sending to Groq model...")

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        raise Exception(f"Error communicating with Groq: {str(e)}") from e