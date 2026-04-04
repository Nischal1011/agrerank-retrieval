# Create a retrieval-augmented generation (RAG) system using OpenAI's GPT-4o-mini and FAISS

from vectorrag.retrieve import search
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def build_prompt(query, hits):
    context = "\n\n".join([f"[{i+1}] {h['text']}" for i, h in enumerate(hits)])
    return f"""Use ONLY the external context below to answer the question.

External context:
{context}

Question: {query}
Answer:"""

def answer_query(query, k=10):
    hits = search(query, k)   # not top_k
    prompt = build_prompt(query, hits)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content, hits


if __name__ == "__main__":
    q = "Explain graph fusion embedding."
    ans, ctx = answer_query(q, k=5)
    print("Answer:\n", ans)
