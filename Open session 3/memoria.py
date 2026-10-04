"""Memoria semántica de largo plazo en Pinecone.

Se sigue el patrón de la Unidad 7:
- un paso de RECUPERACIÓN al inicio de cada turno (similitud semántica con el
  mensaje del usuario, filtrado por user_id);
- un paso de EXTRACCIÓN separado del que conversa, que decide qué vale la pena
  guardar, con deduplicación contra el recuerdo más parecido.

Los ids de los vectores llevan el user_id como prefijo, así se pueden listar o
borrar todos los recuerdos de un usuario con index.list(prefix=...).
"""
import hashlib
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pinecone import ServerlessSpec
from pydantic import BaseModel, Field

from rag_tools import embedding_model_instance, pinecone_instance

load_dotenv()

INDEX_MEMORIA = os.getenv("PINECONE_MEMORY_INDEX", "semantic-memory")
NAMESPACE = "open-session-3"
DIMENSION = 384  # paraphrase-multilingual-MiniLM-L12-v2

TOP_K_MEMORIA = 4
MIN_SCORE_MEMORIA = 0.2
# A partir de este score se considera que el hecho nuevo habla de lo mismo que
# uno ya guardado y se le pregunta al LLM si es duplicado o actualización.
UMBRAL_DEDUP = 0.6

# Salida estructurada (lista de hechos) en vez de tool calling: con nano es
# más confiable que esperar que "decida" llamar a una tool de guardado.
llm_extraccion = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-nano")
llm_dedup = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-nano")


def _crear_indice_si_falta() -> None:
    # La memoria persistente necesita un índice separado del conocimiento:
    # mezclar ambos dominios haría que una preferencia del usuario apareciera
    # como si fuera un dato del catálogo.
    nombres = [i["name"] for i in pinecone_instance.list_indexes()]
    if INDEX_MEMORIA not in nombres:
        print(f"[memoria] creando índice '{INDEX_MEMORIA}' en Pinecone...")
        pinecone_instance.create_index(
            name=INDEX_MEMORIA,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


_crear_indice_si_falta()
index = pinecone_instance.Index(INDEX_MEMORIA)


def _id_memoria(user_id: str, texto: str) -> str:
    """Mismo usuario + mismo texto normalizado => mismo id (upsert idempotente)."""
    digest = hashlib.sha256(texto.strip().lower().encode()).hexdigest()[:32]
    return f"{user_id}#{digest}"


def _embed(texto: str) -> list:
    # La consulta y el hecho usan el mismo espacio vectorial para que la
    # recuperación semántica compare significado y no coincidencia literal.
    return embedding_model_instance.encode(texto).tolist()


def _buscar(user_id: str, texto: str, k: int) -> list:
    resultado = index.query(
        vector=_embed(texto),
        top_k=k,
        namespace=NAMESPACE,
        filter={"user_id": user_id},
        include_metadata=True,
    )
    return resultado.get("matches", [])


# ---------------------------------------------------------------- lectura

def recuperar_memorias(user_id: str, consulta: str) -> list[str]:
    """Hechos del usuario relevantes para la consulta actual."""
    # Recuperar antes de razonar permite que el supervisor incorpore contexto
    # persistente sin confundirlo con el historial del thread actual.
    return [
        m["metadata"]["texto"]
        for m in _buscar(user_id, consulta, TOP_K_MEMORIA)
        if m["score"] >= MIN_SCORE_MEMORIA
    ]


def _ids_usuario(user_id: str) -> list[str]:
    ids = []
    for lote in index.list(prefix=f"{user_id}#", namespace=NAMESPACE):
        # Según la versión del cliente, cada item es un str o un ListItem(id=...).
        ids.extend(getattr(item, "id", item) for item in lote)
    return ids


def listar_memorias(user_id: str) -> list[str]:
    # list/fetch se usa para la vista administrativa completa, no para el
    # ranking semántico que alimenta una respuesta puntual.
    ids = _ids_usuario(user_id)
    if not ids:
        return []
    vectores = index.fetch(ids=ids, namespace=NAMESPACE).vectors
    return [v.metadata["texto"] for v in vectores.values()]


def borrar_memorias(user_id: str) -> int:
    # El prefijo del ID funciona como partición lógica por usuario y permite
    # borrar sus recuerdos sin afectar a otros usuarios.
    ids = _ids_usuario(user_id)
    if ids:
        index.delete(ids=ids, namespace=NAMESPACE)
    return len(ids)


# -------------------------------------------------------------- escritura

class HechosExtraidos(BaseModel):
    hechos: list[str] = Field(
        default_factory=list,
        description=(
            "Hechos estables sobre el usuario, cada uno autocontenido y en "
            "tercera persona (ej: 'Al usuario le gustan las películas de "
            "ciencia ficción'). Lista vacía si no hay nada que recordar."
        ),
    )


PROMPT_EXTRACCION = """Sos el paso de EXTRACCIÓN DE MEMORIA de un asistente. No le respondés al
usuario: tu único trabajo es decidir si su último mensaje contiene algo que
valga la pena recordar en FUTURAS conversaciones.

Guardá solo hechos estables que el usuario AFIRMA sobre sí mismo: nombre,
dónde vive, a qué se dedica, gustos y preferencias (géneros de películas o
libros, dieta, unidades, idioma), restricciones (alergias, vegetariano), datos
que podría volver a necesitar (su ciudad, su país).

NO guardes:
- preguntas ("¿qué tiempo hace en Madrid?" no dice nada de la persona);
- pedidos puntuales de este momento ("buscame una receta");
- descripciones del mensaje o de la conversación ("el usuario saluda");
- nada inferido o supuesto. Ante la duda, no guardes nada.

Cada hecho debe poder leerse solo dentro de un año, redactado en tercera
persona y en español. Si varios datos son del mismo tipo, combinalos en un
único hecho ("Al usuario le gustan la ciencia ficción y el terror")."""


def _clasificar_relacion(nuevo: str, existente: str) -> str:
    # La similitud vectorial encuentra candidatos; el LLM decide si el texto es
    # duplicado, contradicción, actualización o un hecho independiente.
    prompt = (
        "Compará dos afirmaciones sobre el mismo usuario y respondé con una sola "
        "palabra: DUPLICATE, CONTRADICTION, UPDATE o NEW.\n"
        "DUPLICATE: dicen lo mismo aunque estén redactadas distinto.\n"
        "CONTRADICTION: la nueva invierte o reemplaza a la existente.\n"
        "UPDATE: la nueva amplía o precisa a la existente sin contradecirla.\n"
        "NEW: no tienen relación.\n\n"
        f"Existente: {existente}\nNueva: {nuevo}"
    )
    respuesta = llm_dedup.invoke([SystemMessage(content=prompt)]).content.upper()
    for etiqueta in ("DUPLICATE", "CONTRADICTION", "UPDATE"):
        if etiqueta in respuesta:
            return etiqueta
    return "NEW"


def _guardar_hecho(user_id: str, hecho: str) -> str:
    """Guarda un hecho deduplicando contra el recuerdo más parecido del usuario."""
    # Primero se busca un único vecino: solo si supera UMBRAL_DEDUP se invoca
    # una clasificación más costosa para decidir si reemplazarlo o conservarlo.
    cercanos = _buscar(user_id, hecho, 1)
    if cercanos and cercanos[0]["score"] >= UMBRAL_DEDUP:
        previo = cercanos[0]
        relacion = _clasificar_relacion(hecho, previo["metadata"]["texto"])
        if relacion == "DUPLICATE":
            return f"omitido (duplicado de '{previo['metadata']['texto']}')"
        if relacion in ("CONTRADICTION", "UPDATE"):
            index.delete(ids=[previo["id"]], namespace=NAMESPACE)
            accion = f"reemplaza a '{previo['metadata']['texto']}' ({relacion})"
        else:
            accion = "nuevo"
    else:
        accion = "nuevo"

    index.upsert(
        vectors=[{
            "id": _id_memoria(user_id, hecho),
            "values": _embed(hecho),
            "metadata": {"user_id": user_id, "texto": hecho},
        }],
        namespace=NAMESPACE,
    )
    return f"guardado: {accion}"


def extraer_y_guardar(user_id: str, mensaje_usuario: str) -> list[str]:
    """Paso dedicado de extracción: decide qué guardar y lo persiste.

    Devuelve líneas de log ('hecho -> resultado') para la auditoría.
    """
    # Separar extracción y conversación evita que el LLM que responde al usuario
    # tenga que decidir simultáneamente qué información persistir.
    extraccion = llm_extraccion.with_structured_output(HechosExtraidos).invoke([
        SystemMessage(content=PROMPT_EXTRACCION),
        ("human", f"Último mensaje del usuario: {mensaje_usuario}"),
    ])
    return [f"'{h}' -> {_guardar_hecho(user_id, h)}" for h in extraccion.hechos if h.strip()]
