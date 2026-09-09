# Open Session 2 — Agente multidominio

Agente que responde consultas sobre **películas**, **libros** y **recetas**
usando RAG contra bases vectoriales, y sobre **clima** y **países** usando
herramientas que consultan APIs públicas en vivo. Construido con
[LangGraph](https://langchain-ai.github.io/langgraph/) siguiendo el patrón
**ReAct**, armado "a mano" con `StateGraph` — el mismo enfoque que usa el
profe en `Unidad 6 - Agents and tools` (demo2 a demo7), no el helper
`create_react_agent` de más alto nivel.

## Arquitectura

```
Usuario
  │
  ▼
┌─────────────────────────────┐
│   nodo "agent" (LLM + tools bindeadas)  │◄──┐
└─────────────────────────────┘   │
  │ tools_condition               │
  ▼                               │
┌─────────────────────────────┐   │
│   nodo "tools" (ToolNode)    │───┘
└─────────────────────────────┘
  │
  ▼
buscar_peliculas   ──► índice Pinecone "open-session-2-peliculas"  (RAG)
buscar_libros      ──► índice Pinecone "open-session-2-libros"     (RAG)
buscar_recetas     ──► índice Pinecone "open-session-2-recetas"    (RAG)
consultar_clima    ──► Open-Meteo (geocoding + forecast)           (API pública)
consultar_pais     ──► RestCountries v3.1                          (API pública)
```

`agent.py` arma exactamente el mismo grafo que `demo7-agent-sql-charts.py`
de la Unidad 6: un nodo `agent` que llama al LLM con las tools bindeadas
(`llm.bind_tools(...)`), un nodo `tools` con `ToolNode`, y una arista
condicional (`tools_condition`) que decide si hace falta ejecutar una tool
o si ya se puede terminar. El LLM elige sola la(s) tool(s) según la
pregunta — no hay un clasificador manual como en el chatbot de países de
Open session 1. Para una consulta combinada ("¿cómo está el clima en la
capital de Francia?"), el grafo pasa dos veces por el nodo `tools`: primero
`consultar_pais` para obtener la capital, y con ese resultado
`consultar_clima`.

## ¿Por qué hay JSON en `data/` si la letra pide "base vectorial"?

Los archivos `data/books.json` y `data/recipes.json` **no son la base de
conocimiento del agente** — son el dataset crudo de origen, el mismo rol
que cumple `imdb-top-1000.csv` de la Unidad 5 para películas. `ingest.py`
los lee una sola vez, genera los embeddings y los sube a Pinecone. En
tiempo de ejecución, `tools_domain.py` (las tools que usa el agente)
**nunca lee esos JSON**: hace `similarity_search` directo contra el índice
de Pinecone. La recuperación real es 100% vía base vectorial, tal como pide
la letra — el JSON es solo de dónde salen los datos antes de vectorizarse.

## Decisiones de diseño

- **Patrón del agente**: `StateGraph` manual (`agent` + `tools` +
  `tools_condition`), igual que `Unidad 6/demo7-agent-sql-charts.py` —
  no `langgraph.prebuilt.create_react_agent` ni `langchain.agents.create_agent`
  (esos son atajos de más alto nivel que usan las demos 1 y 2 de la Unidad
  6 antes de mostrar el patrón "a mano").
- **Versiones de librerías**: `requirements.txt` fija las mismas versiones
  de `langchain` / `langchain-openai` / `langchain-core` / `langgraph` que
  `Unidad 6/requirements.txt`, para evitar incompatibilidades con el resto
  del código del curso.
- **Embeddings para RAG**: `sentence-transformers/all-MiniLM-L6-v2` vía
  `langchain_huggingface`, y `langchain_pinecone.PineconeVectorStore` para
  el vectorstore — el mismo combo que usó el profe en
  `Unidad 5/demo5-vector-database.py` y `demo6-update-vector-database.py`.
- **Tres índices separados** (uno por dominio) en vez de uno solo con un
  campo `domain` en la metadata: así cada tool busca solo en su propio
  dominio y no hace falta filtrar resultados de otro tema.
- **Dataset de películas**: se reutiliza `imdb-top-1000.csv` de la
  Unidad 5 (primeras 300 filas, alcanza para los casos de prueba —
  incluye Inception y Titanic).
- **Datasets de libros y recetas**: no venían provistos, así que se armaron
  a mano en `data/books.json` y `data/recipes.json` (20 ítems cada uno,
  cubriendo los ejemplos de la letra: Cien años de soledad, El Hobbit,
  Orgullo y prejuicio, lasaña, guacamole, brownies, opciones vegetarianas)
  — ver la aclaración arriba sobre por qué esto sigue siendo 100% RAG.
- **Tool de países**: la demo de Open session 1 (`paises-api.js`) usa
  `api.restcountries.com/v5`, que es de pago y pide una
  `RESTCOUNTRIES_API_KEY`. Para no depender de esa key usamos la API
  pública y gratuita **RestCountries v3.1** (`restcountries.com/v3.1`),
  que da los mismos datos que pide la letra (capital, moneda, idiomas,
  población).
- **Tool de clima**: misma lógica que `weather_tool` de
  `Unidad 6/demo3-agent-tools.py` (geocoding + forecast de Open-Meteo, con
  el agregado de traducir el `weathercode` a una descripción en español,
  como hacía `Unidad 3/weather-api/server.js`).
- **Manejo de incertidumbre**: va en el `SYSTEM_PROMPT` del agente
  (`agent.py`), no en código — se le pide explícitamente que no responda
  con conocimiento general para los 5 dominios (todo tiene que salir de
  una tool) y que reconozca cuando no tiene información suficiente.

## Cómo correrlo

1. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Copiar `.env.example` a `.env` y completar las API keys
   (`OPENAI_API_KEY`, `PINECONE_API_KEY`, `HF_TOKEN`).

3. Armar los índices vectoriales (una sola vez):

   ```bash
   python ingest.py
   ```

4. Probar el agente:

   ```bash
   python chat.py          # chat interactivo por consola (respuesta final)
   python test_cases.py    # corre los casos de prueba de la letra, mostrando
                            # qué tool eligió el agente en cada paso
   ```

## Qué falta / posibles mejoras

- El agente no mantiene memoria entre preguntas dentro de `chat.py` (cada
  `preguntar()` es un turno nuevo) — es el mismo comportamiento que las
  demos de consola de la Unidad 6. Para historial multi-turno habría que
  agregar un checkpointer de LangGraph (`MemorySaver`); no lo pide la
  letra, pero es la extensión natural.
- `buscar_*` siempre trae `k=3` resultados; para catálogos más grandes
  convendría exponer `top_k` como parámetro de la tool.
- No hay búsqueda híbrida (metadata + semántica) como en el ejercicio de
  FerreMax del Práctico 4 — no la pide esta letra, pero el patrón ya está
  armado en `demo2-vectorial-search.py` de la Unidad 4 si hiciera falta.
