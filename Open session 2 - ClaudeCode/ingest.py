"""
Arma los tres índices de Pinecone que usan las tools de RAG del agente
(películas, libros y recetas), siguiendo el mismo patrón que usó el profe
en demo5-vector-database.py y demo6-update-vector-database.py de la
Unidad 5: langchain_huggingface para los embeddings + langchain_pinecone
para el vectorstore.

Correrlo una sola vez (o cada vez que cambien los datasets en data/):

    python ingest.py
"""

import csv
import json
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

from config import EMBEDDING_MODEL_NAME, INDEX_LIBROS, INDEX_PELICULAS, INDEX_RECETAS

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_DIMENSION = 384  # dimensión de salida de all-MiniLM-L6-v2

# Ruta al dataset de películas ya usado en la Unidad 5 (no lo duplicamos)
MOVIES_CSV_PATH = os.path.join(
    "..", "Unidad 5 - Pipeline knowledge base", "imdb-top-1000.csv"
)
MOVIES_LIMIT = 300  # alcanza para cubrir los casos de prueba y no demorar el embedding

pc = Pinecone(api_key=PINECONE_API_KEY)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def crear_indice_si_no_existe(index_name):
    if index_name not in [i["name"] for i in pc.list_indexes()]:
        print(f"Creando índice '{index_name}' en Pinecone...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"El índice '{index_name}' ya existe.")


# region Películas

def cargar_documentos_peliculas():
    documentos = []
    with open(MOVIES_CSV_PATH, newline="", encoding="utf-8") as f:
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


# endregion

# region Libros

def cargar_documentos_libros():
    with open(os.path.join("data", "books.json"), encoding="utf-8") as f:
        libros = json.load(f)

    documentos = []
    for libro in libros:
        texto = (
            f"Título: {libro['title']}\n"
            f"Autor: {libro['author']}\n"
            f"Año: {libro['year']}\n"
            f"Género: {libro['genre']}\n"
            f"Sinopsis: {libro['synopsis']}"
        )
        documentos.append(
            Document(
                page_content=texto,
                metadata={
                    "domain": "libros",
                    "title": libro["title"],
                    "author": libro["author"],
                    "year": libro["year"],
                    "genre": libro["genre"],
                },
            )
        )
    return documentos


# endregion

# region Recetas

def cargar_documentos_recetas():
    with open(os.path.join("data", "recipes.json"), encoding="utf-8") as f:
        recetas = json.load(f)

    documentos = []
    for receta in recetas:
        texto = (
            f"Receta: {receta['name']}\n"
            f"Categoría: {receta['category']}\n"
            f"Ingredientes: {', '.join(receta['ingredients'])}\n"
            f"Preparación: {receta['steps']}"
        )
        documentos.append(
            Document(
                page_content=texto,
                metadata={
                    "domain": "recetas",
                    "name": receta["name"],
                    "category": receta["category"],
                },
            )
        )
    return documentos


# endregion


def indexar(index_name, documentos):
    crear_indice_si_no_existe(index_name)
    index = pc.Index(index_name)
    stats = index.describe_index_stats()

    if stats["total_vector_count"] > 0:
        print(f"'{index_name}' ya tiene {stats['total_vector_count']} vectores, no se reindexa.")
        print("(Si cambiaste el dataset, borrá el índice en Pinecone y volvé a correr este script.)")
        return

    print(f"Generando embeddings y subiendo {len(documentos)} documentos a '{index_name}'...")
    PineconeVectorStore.from_documents(
        documents=documentos,
        embedding=embedding_model,
        index_name=index_name,
    )
    print(f"'{index_name}' listo.")


if __name__ == "__main__":
    indexar(INDEX_PELICULAS, cargar_documentos_peliculas())
    indexar(INDEX_LIBROS, cargar_documentos_libros())
    indexar(INDEX_RECETAS, cargar_documentos_recetas())

    print("\nÍndices listos. Ahora podés correr: python chat.py")
