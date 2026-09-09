"""Constantes compartidas entre ingest.py, tools_domain.py y agent.py."""

import os

# Nombres de los índices de Pinecone (uno por dominio de RAG)
INDEX_PELICULAS = "open-session-2-peliculas"
INDEX_LIBROS = "open-session-2-libros"
INDEX_RECETAS = "open-session-2-recetas"

# Mismo modelo de embeddings que usaron todas las demos de RAG del curso
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Modelo de OpenAI para el agente (configurable por .env)
MODELO = os.getenv("MODELO", "gpt-4.1-nano")
