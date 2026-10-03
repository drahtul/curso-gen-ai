"""Tools de RAG del subagente de conocimiento (base vectorial propia en Pinecone).

Un único índice con un namespace por dominio (peliculas, libros, recetas). Se
carga con `python ingesta_rag.py` a partir de data/conocimiento.json.
"""
import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

# Se usa paraphrase-multilingual-MiniLM-L12-v2 en lugar de all-MiniLM-L6-v2 porque toda la informacion está en español.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
INDEX_CONOCIMIENTO = os.getenv("PINECONE_KNOWLEDGE_INDEX", "os3-conocimiento")

TOP_K = 4
MIN_SCORE = 0.2

SIN_INFO = (
    "SIN_RESULTADOS: la base vectorial de {dominio} no contiene información "
    "relevante para esta consulta."
)

embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL)
pinecone_instance = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

ETIQUETAS = (
    ("year", "año"),
    ("director", "director"),
    ("author", "autor"),
    ("genre", "género"),
    ("duration", "duración"),
    ("type", "tipo"),
    ("ingredients", "ingredientes"),
)


def _format_match(match: dict) -> str:
    meta = match.get("metadata") or {}
    extras = " | ".join(f"{etiqueta}: {meta[clave]}" for clave, etiqueta in ETIQUETAS if meta.get(clave))
    partes = [f"Título: {meta.get('title', 'Sin título')}"]
    if extras:
        partes.append(extras)
    partes.append(f"Contenido: {meta.get('text', '').strip()}")
    partes.append(f"(similitud: {match['score']:.3f})")
    return "\n".join(partes)


def buscar_en_indice(consulta: str, namespace: str, dominio: str) -> str:
    """Búsqueda semántica en un namespace del índice de conocimiento.

    Devuelve SIN_RESULTADOS si el índice no existe o ningún resultado supera el
    umbral: el subagente usa esa señal para admitir que no tiene el dato.
    """
    if INDEX_CONOCIMIENTO not in [i["name"] for i in pinecone_instance.list_indexes()]:
        return SIN_INFO.format(dominio=dominio) + (
            f" (el índice '{INDEX_CONOCIMIENTO}' no existe: correr ingesta_rag.py)"
        )

    resultados = pinecone_instance.Index(INDEX_CONOCIMIENTO).query(
        vector=embedding_model_instance.encode(consulta).tolist(),
        top_k=TOP_K,
        namespace=namespace,
        include_metadata=True,
    )
    matches = [m for m in resultados.get("matches", []) if m["score"] >= MIN_SCORE]
    if not matches:
        return SIN_INFO.format(dominio=dominio)

    encabezado = (
        f"Resultados de la base vectorial de {dominio} para '{consulta}'. "
        "Usá EXCLUSIVAMENTE estos datos. Si el dato puntual que pidió el usuario "
        "no está escrito acá, decí que no lo tenés: está prohibido completarlo "
        "con conocimiento propio."
    )
    return encabezado + "\n\n" + "\n\n".join(_format_match(m) for m in matches)


@tool
def buscar_peliculas(consulta: str) -> str:
    """Busca PELÍCULAS en la base vectorial de cine del catálogo.

    Sirve para recomendaciones por tema o género, director, año, duración o
    sinopsis. Ejemplos: 'películas de ciencia ficción en el espacio',
    'director de Interstellar', 'películas argentinas'.
    """
    return buscar_en_indice(consulta, "peliculas", "películas")


@tool
def buscar_libros(consulta: str) -> str:
    """Busca LIBROS en la base vectorial de literatura del catálogo.

    Sirve para recomendaciones, autor, año de publicación, género o sinopsis.
    Ejemplos: 'quién escribió Rayuela', 'novelas de ciencia ficción'.
    """
    return buscar_en_indice(consulta, "libros", "libros")


@tool
def buscar_recetas(consulta: str) -> str:
    """Busca RECETAS de cocina en la base vectorial de recetas.

    Sirve para preparaciones, ingredientes o recomendaciones de platos. Incluí
    en la consulta restricciones de dieta si las hay. Ejemplos: 'cena
    vegetariana', 'cómo preparar lasaña', 'postres'.
    """
    return buscar_en_indice(consulta, "recetas", "recetas")
