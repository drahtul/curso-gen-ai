from sentence_transformers import SentenceTransformer # type: ignore
import os
from dotenv import load_dotenv
from huggingface_hub import login
from pinecone import Pinecone # type: ignore
from openai import OpenAI

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "obli-rag-cine"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query, top_k=5):
    query_embedding = model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return results["matches"]

def ask_llm_movie_recommendation(user_query, retrieved_docs):
    context = "\n\n".join(
        [
            f"Title: {doc['metadata'].get('title')}\n"
            f"Text: {doc['metadata'].get('text')}\n"
            f"Rating: {doc['metadata'].get('rating')}\n"
            f"Year: {doc['metadata'].get('year')}"
            for doc in retrieved_docs
        ]
    )

    response = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a movie recommendation assistant. "
                    "Use ONLY the provided context to recommend movies. "
                    "Explain briefly why each movie fits the user request."
                )
            },
            {
                "role": "user",
                "content": f"""
User query: {user_query}

Context (movies retrieved from vector database):
{context}

Return a ranked list of 3 recommendations with explanation.
"""
            }
        ]
    )

    return response.output_text

query = "I want comedy movies recommendations"

docs = semantic_search(
    query=query
)

answer = ask_llm_movie_recommendation(query, docs)

print("\n🎬 MOVIE RECOMMENDATIONS\n")
print(answer)