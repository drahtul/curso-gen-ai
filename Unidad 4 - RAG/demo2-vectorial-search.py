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


def semantic_search(query, top_k=1):
    # La consulta debe vectorizarse con el mismo modelo usado al indexar los
    # documentos para que los vectores vivan en el mismo espacio semantico.
    query_embedding = model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    # top_k controla cuantos candidatos devuelve Pinecone; el score depende
    # de la metrica configurada en el indice.
    # print(query_embedding)
    print("\n🔍 SEMANTIC SEARCH RESULTS")
    for match in results["matches"]:
        print(f"- Score: {match['score']:.4f}")
        print(f"  Text: {match['metadata'].get('text')}")
        print(f"  Metadata: {match['metadata']}")
        print()


def hybrid_search(query, rating=None, year=None, top_k=5):
    query_embedding = model.encode(query).tolist()

    # En este ejemplo "hibrida" significa similitud vectorial mas filtros de
    # metadata; no combina embeddings con una busqueda lexical como BM25.
    filter_dict = {}

    if rating:
        filter_dict["rating"] = {"$eq": rating}

    if year:
        # year se almaceno como texto durante la ingestion, por eso el filtro
        # convierte el valor recibido a string y exige igualdad exacta.
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
        print(f"  title: {match['metadata'].get('title')}")
        print(f"  Text: {match['metadata'].get('text')}")
        print(f"  Metadata: {match['metadata']}")
        print()



# query = "movie that is a science fiction thriller with a strong female protagonist"
# escribir un query que tenga relación con la patagonia Argentina
query = "movie set in the argentine patagonia"

semantic_search(query)

hybrid_search(
    query=query,
    rating=6.7,
    year=2009
)