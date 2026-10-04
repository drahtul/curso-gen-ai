El `score` es la **medida de similitud entre la consulta y el documento encontrado**.

En el código se imprime aquí:

`demo2-vectorial-search.py`

```python
print(f"- Score: {match['score']:.4f}")
```

### ¿De dónde sale?

1. La consulta se transforma en un embedding:

```python
query_embedding = model.encode(query).tolist()
```

Por ejemplo:

```text
"movie set in the argentine patagonia"
```

se convierte en un vector de 384 números usando el modelo:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

2. Pinecone compara ese vector con los vectores almacenados previamente para cada película:

```python
results = index.query(
    vector=query_embedding,
    top_k=top_k,
    include_metadata=True
)
```

3. Pinecone calcula la similitud y devuelve cada resultado con su campo:

```python
match["score"]
```

En este proyecto el índice está configurado con la métrica **cosine**, por lo que conceptualmente calcula la similitud coseno:

```text
score = coseno(embedding_de_la_consulta, embedding_de_la_película)
```

La similitud coseno compara el ángulo entre ambos vectores:

- `1`: vectores muy similares.
- `0`: poca relación semántica.
- `-1`: relación opuesta, aunque normalmente los embeddings de este modelo producen valores positivos o cercanos a positivos.

### Importante

El `score` **no es una probabilidad ni un porcentaje de certeza**. Un valor como `0.7342` significa que ese documento es más parecido semánticamente a la consulta que otro con `0.4210`, pero no significa que haya un `73.42 %` de probabilidad de que sea correcto.

Además:

```python
top_k=5
```

indica cuántos resultados devuelve Pinecone. El resultado se ordena normalmente del `score` más alto al más bajo.

En `hybrid_search`, los filtros:

```python
rating=6.7
year=2009
```

primero restringen los documentos candidatos mediante metadata. Luego Pinecone calcula el `score` vectorial únicamente sobre esos candidatos. Por eso, los filtros no modifican directamente el score; modifican qué documentos pueden aparecer en los resultados.