from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("Loaded API Key:", api_key) 

client = OpenAI(api_key=api_key)


def generate_answer(query, retrieved_chunks):

    context = "\n\n".join([chunk[:500] for chunk in retrieved_chunks])

    prompt = f"""
Answer the question using ONLY the context.

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content