from sentence_transformers import SentenceTransformer # type: ignore
import os
from dotenv import load_dotenv
from huggingface_hub import login
from pinecone import Pinecone # type: ignore

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "obli-rag-cine"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_search(query, top_k=10):
    query_embedding = model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    print("\n🔍 SEMANTIC SEARCH RESULTS")
    for match in results["matches"]:
        print(f"- Score: {match['score']:.4f}")
        print(f"  Text: {match['metadata'].get('text')}")
        print(f"  Metadata: {match['metadata']}")
        print()


def hybrid_search(query, rating=None, year=None, top_k=5):
    query_embedding = model.encode(query).tolist()

    # Construcción del filtro metadata
    filter_dict = {}

    if rating:
        filter_dict["rating"] = {"$eq": rating}

    if year:
        filter_dict["year"] = {"$eq": f"{year}"}

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None
    )

    print("\n🧩 HYBRID SEARCH RESULTS")
    print("Filtro aplicado:", filter_dict)

    for match in results["matches"]:
        print(f"- Score: {match['score']:.4f}")
        print(f"  Text: {match['metadata'].get('text')}")
        print(f"  Metadata: {match['metadata']}")
        print()



query = "action movie"

semantic_search(query)

hybrid_search(
    query=query,
    rating=6.7,
    year=2016
)