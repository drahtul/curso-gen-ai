import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_MOVIES = "movies-index"
INDEX_BOOKS = "books-index"
INDEX_RECIPES = "recipes-index"

TOP_K = 5
MIN_SCORE = 0.25

SIN_INFO = (
    "SIN_RESULTADOS: la base vectorial de {dominio} no contiene información "
    "relevante para esta consulta."
)

embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL)
pinecone_instance = Pinecone(api_key=PINECONE_API_KEY)
# Todas las consultas deben usar el mismo modelo de embeddings que se uso al
# indexar, para vivir en el mismo espacio vectorial.


def _index_exists(index_name: str) -> bool:
    return index_name in [i["name"] for i in pinecone_instance.list_indexes()]


def _format_match(match: dict) -> str:
    """Arma una línea legible con los metadatos disponibles del chunk."""
    meta = match.get("metadata") or {}
    titulo = meta.get("title") or meta.get("titulo") or meta.get("name") or "Sin título"

    extras = []
    for clave, etiqueta in (
        ("year", "año"),
        ("author", "autor"),
        ("genres", "géneros"),
        ("genre", "género"),
        ("rating", "rating"),
        ("type", "tipo"),
        ("ingredients", "ingredientes"),
    ):
        if meta.get(clave):
            extras.append(f"{etiqueta}: {meta[clave]}")

    texto = (meta.get("text") or meta.get("content") or "").strip()

    partes = [f"Título: {titulo}"]
    if extras:
        partes.append(" | ".join(extras))
    if texto:
        partes.append(f"Contenido: {texto}")
    partes.append(f"(similitud: {match['score']:.3f})")
    return "\n".join(partes)


def buscar_en_indice(consulta: str, index_name: str, dominio: str) -> str:
    """Búsqueda semántica genérica sobre un índice de Pinecone.

    Devuelve SIN_RESULTADOS si el índice no existe, está vacío o ningún match
    supera el umbral de similitud; el agente usa esa señal para admitir que no
    tiene información en lugar de inventarla.
    """
    # Retrieval: se transforma la consulta en vector y se recuperan candidatos
    # por similitud semantica antes de que el LLM redacte una respuesta.
    if not _index_exists(index_name):
        return SIN_INFO.format(dominio=dominio) + f" (el índice '{index_name}' no existe)"

    index = pinecone_instance.Index(index_name)
    vector = embedding_model_instance.encode(consulta).tolist()

    resultados = index.query(
        vector=vector,
        top_k=TOP_K * 3,  # se pide de más porque luego se deduplica
        include_metadata=True,
    )

    # Se filtra por umbral y se deduplican obras porque varios chunks pueden
    # pertenecer al mismo titulo; asi el contexto final evita repeticiones.
    matches = []
    titulos_vistos = set()
    for m in resultados.get("matches", []):
        if m["score"] < MIN_SCORE:
            continue
        # Los indices suelen tener varios chunks de la misma obra: se conserva
        # el mejor de cada titulo para no repetir informacion.
        titulo = ((m.get("metadata") or {}).get("title") or m["id"]).lower()
        if titulo in titulos_vistos:
            continue
        titulos_vistos.add(titulo)
        matches.append(m)

    if not matches:
        return SIN_INFO.format(dominio=dominio)

    # El encabezado convierte la recuperacion en una instruccion de grounding:
    # si el dato no aparece literalmente, el agente debe reconocer la ausencia.
    encabezado = (
        f"Resultados de la base vectorial de {dominio} para '{consulta}'. "
        "Usá EXCLUSIVAMENTE los datos que aparecen abajo. Si el dato puntual que "
        "pidió el usuario (director, autor, duración, ingredientes, etc.) no está "
        "escrito literalmente en estos resultados, decí que no lo tenés: está "
        "prohibido completarlo con conocimiento propio."
    )
    return encabezado + "\n\n" + "\n\n".join(_format_match(m) for m in matches)


@tool
def buscar_peliculas(consulta: str) -> str:
    """Busca información sobre PELÍCULAS y series en la base vectorial de cine.

    Usar para recomendaciones, directores, actores, duración, año, género o
    sinopsis de una película. El índice está indexado en inglés, así que la
    consulta conviene formularla en inglés: 'science fiction movies',
    'Inception director', 'Titanic runtime'.
    """
    return buscar_en_indice(consulta, INDEX_MOVIES, "películas")


@tool
def buscar_libros(consulta: str) -> str:
    """Busca información sobre LIBROS en la base vectorial de literatura.

    Usar para recomendaciones, autores, año de publicación, género o sinopsis
    de un libro. Por ejemplo: 'quién escribió Cien años de soledad' o
    'libros de fantasía'.
    """
    return buscar_en_indice(consulta, INDEX_BOOKS, "libros")


@tool
def buscar_recetas(consulta: str) -> str:
    """Busca RECETAS de cocina en la base vectorial de recetas.

    Usar para preparaciones, ingredientes, pasos o recomendaciones de platos.
    Por ejemplo: 'cómo preparar lasaña' o 'recetas vegetarianas'.
    """
    return buscar_en_indice(consulta, INDEX_RECIPES, "recetas")
