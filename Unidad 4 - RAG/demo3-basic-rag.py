from sentence_transformers import SentenceTransformer # type: ignore
import os
from dotenv import load_dotenv
from huggingface_hub import login
from pinecone import Pinecone # type: ignore
from openai import OpenAI

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "imbd-top-1000"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query, top_k=5):
    # Retrieval: se recuperan documentos semanticamente cercanos a la consulta
    # para aportar al modelo informacion relacionada con la pregunta.
    query_embedding = model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return results["matches"]

def ask_llm_movie_recommendation(user_query, retrieved_docs):
    # Augmentation: los resultados recuperados se convierten en un contexto
    # legible que luego se incorpora al prompt enviado al LLM.
    context = "\n\n".join(
        [
            f"Title: {doc['metadata'].get('title')}\n"
            f"Text: {doc['metadata'].get('text')}\n"
            f"Rating: {doc['metadata'].get('rating')}\n"
            f"Year: {doc['metadata'].get('year')}"
            for doc in retrieved_docs
        ]
    )

    # El LLM no consulta Pinecone directamente: solo recibe el contexto que
    # este programa construyo a partir de los documentos recuperados.
    # print(context)
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

# Generation: el modelo redacta la respuesta usando los cinco documentos
# recuperados, aunque el prompt solicita seleccionar tres recomendaciones.
answer = ask_llm_movie_recommendation(query, docs)

print("\n🎬 MOVIE RECOMMENDATIONS\n")
print(answer)