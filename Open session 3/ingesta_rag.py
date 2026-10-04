"""Carga data/conocimiento.json en el índice de Pinecone del subagente de conocimiento.

    python ingesta_rag.py

Es idempotente: los ids salen del dominio + título, así que volver a correrlo
actualiza los documentos en lugar de duplicarlos.
"""
import json
import re
import time
from pathlib import Path

from pinecone import ServerlessSpec

from rag_tools import INDEX_CONOCIMIENTO, embedding_model_instance, pinecone_instance

DATA = Path(__file__).parent / "data" / "conocimiento.json"


def _slug(texto: str) -> str:
    # El slug produce identificadores legibles y estables para que la ingesta
    # sea repetible sin crear vectores duplicados.
    return re.sub(r"[^a-z0-9]+", "-", texto.lower()).strip("-")


def _texto_a_embeber(doc: dict) -> str:
    """Se embebe el título + metadatos + sinopsis: así una búsqueda por
    director, género o 'vegetariana' también encuentra el documento."""
    # Incluir metadatos mejora la recuperación cuando la consulta menciona un
    # año, autor, género o ingrediente y no solamente la sinopsis.
    campos = [doc["title"]] + [str(v) for k, v in doc.items() if k not in ("title", "text")] + [doc["text"]]
    return ". ".join(campos)


def main() -> None:
    # Primero se garantiza la infraestructura; después se transforma cada
    # documento en un vector y se publica en el namespace de su dominio.
    if INDEX_CONOCIMIENTO not in [i["name"] for i in pinecone_instance.list_indexes()]:
        print(f"Creando índice '{INDEX_CONOCIMIENTO}'...")
        pinecone_instance.create_index(
            name=INDEX_CONOCIMIENTO,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pinecone_instance.describe_index(INDEX_CONOCIMIENTO).status["ready"]:
            # Pinecone crea el índice de forma asíncrona: consultar ready evita
            # enviar upserts antes de que exista capacidad de escritura.
            time.sleep(2)

    index = pinecone_instance.Index(INDEX_CONOCIMIENTO)
    for namespace, docs in json.loads(DATA.read_text(encoding="utf-8")).items():
        vectores = [
            {
                "id": f"{namespace}-{_slug(d['title'])}",
                "values": embedding_model_instance.encode(_texto_a_embeber(d)).tolist(),
                "metadata": {k: str(v) for k, v in d.items()},
            }
            for d in docs
        ]
        index.upsert(vectors=vectores, namespace=namespace)
        print(f"  {namespace}: {len(vectores)} documentos")

    print("Ingesta terminada.")


if __name__ == "__main__":
    main()
