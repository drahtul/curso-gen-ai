import csv
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore # type: ignore - python 3.12
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# region Extraccion de datos del CSV

csv_path = "imdb-top-1000.csv"

# endregion

# region Creacion de documentos

MOVIES_LIMIT = 1000

def cargar_documentos_peliculas():
    documentos = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in list(reader)[:MOVIES_LIMIT]:
            texto = (
                f"Título: {row['Series_Title']}\n"
                f"Año: {row['Released_Year']}\n"
                f"Género: {row['Genre']}\n"
                f"Director: {row['Director']}\n"
                f"Duración: {row['Runtime']}\n"
                f"Sinopsis: {row['Overview']}"
            )
            documentos.append(
                Document(
                    page_content=texto,
                    metadata={
                        "domain": "peliculas",
                        "title": row["Series_Title"],
                        "year": row["Released_Year"],
                        "genre": row["Genre"],
                        "director": row["Director"],
                        "runtime": row["Runtime"],
                        "rating": row["IMDB_Rating"],
                    },
                )
            )
    return documentos

documentos = cargar_documentos_peliculas()

# endregion

# region Configuracion de Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "imbd-top-1000"

pc = Pinecone(api_key=PINECONE_API_KEY)

if index_name not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

print("Pinecone configuration completed.")

# endregion

# region Carga en la base vectorial

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index = pc.Index(index_name)
index_stats = index.describe_index_stats()

if index_stats["total_vector_count"] > 0:
    vectorstore = PineconeVectorStore.from_existing_index(
        embedding=embedding_model,
        index_name=index_name,
    )
    print("Existing vector index loaded.")
else:
    vectorstore = PineconeVectorStore.from_documents(
        documents=documentos,
        embedding=embedding_model,
        index_name=index_name,
    )
    print(f"Loaded {len(documentos)} documents into Pinecone.")

# endregion