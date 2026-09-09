"""
Tools de RAG del agente: búsqueda semántica sobre los tres índices de
Pinecone armados por ingest.py (películas, libros, recetas).

Mismo mecanismo de recuperación que demo3-basic-rag.py de la Unidad 4:
generar el embedding de la consulta y traer los top_k documentos más
cercanos por similitud de coseno. La diferencia es que acá cada función
queda expuesta como una @tool para que el agente de LangGraph decida
cuándo llamarla (en vez de armar el contexto "a mano" como en la demo).
"""

import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import EMBEDDING_MODEL_NAME, INDEX_LIBROS, INDEX_PELICULAS, INDEX_RECETAS

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def _vectorstore(index_name):
    return PineconeVectorStore.from_existing_index(
        embedding=embedding_model,
        index_name=index_name,
    )


def _formatear_resultados(resultados):
    """Convierte los Document recuperados en texto plano para pasarle al LLM."""
    if not resultados:
        return "No se encontraron resultados en la base de conocimiento."
    return "\n\n---\n\n".join(doc.page_content for doc in resultados)


# Los vectorstores se instancian una sola vez al importar el módulo,
# igual que hacían las demos con `index = pc.Index(...)`.
_peliculas_vs = _vectorstore(INDEX_PELICULAS)
_libros_vs = _vectorstore(INDEX_LIBROS)
_recetas_vs = _vectorstore(INDEX_RECETAS)


@tool
def buscar_peliculas(consulta: str) -> str:
    """Busca películas en la base de conocimiento vectorial. Úsala para
    responder preguntas sobre películas: recomendaciones por género,
    director, duración, año o sinopsis (ej: "recomiéndame una película
    de ciencia ficción", "¿quién dirigió Inception?", "¿cuánto dura
    Titanic?")."""
    print(f"  -> [tool] buscar_peliculas('{consulta}')")
    resultados = _peliculas_vs.similarity_search(consulta, k=3)
    return _formatear_resultados(resultados)


@tool
def buscar_libros(consulta: str) -> str:
    """Busca libros en la base de conocimiento vectorial. Úsala para
    responder preguntas sobre libros: autor, año de publicación, género
    o recomendaciones (ej: "¿quién escribió Cien años de soledad?",
    "recomiéndame un libro de fantasía")."""
    print(f"  -> [tool] buscar_libros('{consulta}')")
    resultados = _libros_vs.similarity_search(consulta, k=3)
    return _formatear_resultados(resultados)


@tool
def buscar_recetas(consulta: str) -> str:
    """Busca recetas de cocina en la base de conocimiento vectorial.
    Úsala para responder cómo preparar un plato, qué ingredientes lleva,
    o para recomendar recetas por categoría (ej: "¿cómo preparo una
    lasaña?", "recomiéndame una receta vegetariana")."""
    print(f"  -> [tool] buscar_recetas('{consulta}')")
    resultados = _recetas_vs.similarity_search(consulta, k=3)
    return _formatear_resultados(resultados)
