# Open Session 3: explicación académica de la solución

## 1. Propósito y alcance

La solución implementa un asistente de información y entretenimiento basado en
un sistema multiagente. El problema exige pasar de un agente único a una
arquitectura de **supervisor con subagentes especialistas**, capaz de:

- resolver dominios diferentes sin mezclar sus reglas;
- usar herramientas propias de la aplicación;
- consumir herramientas de un servidor MCP de terceros;
- recuperar información de una base vectorial mediante RAG;
- conservar el contexto de una conversación;
- recordar hechos relevantes del usuario entre sesiones distintas;
- reconocer explícitamente cuándo no tiene información suficiente.

La implementación organiza el sistema en tres especialistas:

1. **Conocimiento**: películas, libros y recetas mediante RAG.
2. **Datos externos**: clima y datos de países mediante tools propias.
3. **GitHub**: repositorios, código, issues y usuarios mediante MCP.

Esta división no es la única posible, pero es coherente con el requisito de
modularidad: cada subagente conoce solamente su dominio y sus herramientas.

## 2. Correspondencia entre requisitos y componentes

| Requisito | Implementación |
|---|---|
| Supervisor | Nodo `supervisor` en `agente.py` |
| Al menos dos subagentes | `conocimiento`, `datos_externos` y `github` |
| Tool propia | `consultar_clima` y `consultar_pais` en `tools_propias.py` |
| Tool MCP de terceros | Tools de lectura del servidor MCP oficial de GitHub |
| Recuperación RAG | Pinecone + `SentenceTransformer` en `rag_tools.py` |
| Memoria de corto plazo | `InMemorySaver`, indexado por `thread_id` |
| Memoria semántica de largo plazo | Índice Pinecone separado en `memoria.py` |
| Incertidumbre | Marcadores `SIN_RESULTADOS`/`ERROR`, umbrales de similitud y respuesta de fallback |
| Casos de prueba | `casos_prueba.py` |
| Interfaz de ejecución | CLI de `main.py` |

Los archivos de datos del catálogo están en
[`data/conocimiento.json`](./data/conocimiento.json). La carga de ese catálogo
al índice vectorial se realiza con
[`ingesta_rag.py`](./ingesta_rag.py).

## 3. Arquitectura lógica

La arquitectura tiene cuatro capas conceptuales:

### 3.1. Entrada y configuración de sesión

[`main.py`](./main.py) recibe:

- `user_id`: identifica al usuario y permite recuperar sus recuerdos
  persistentes;
- `thread_id`: identifica la conversación actual y permite recuperar el
  historial corto;
- el texto de la consulta.

La distinción es fundamental:

```text
user_id   -> identidad y memoria semántica entre conversaciones
thread_id -> hilo conversacional y memoria de corto plazo
```

Por ejemplo, dos threads diferentes del mismo usuario comparten preferencias
guardadas en Pinecone, pero no necesariamente comparten el historial completo
del thread.

### 3.2. Orquestación

[`agente.py`](./agente.py) construye un `StateGraph` de LangGraph. El estado
central contiene:

- `messages`: historial de mensajes;
- `user_id`: usuario actual;
- `memorias`: recuerdos recuperados para la consulta;
- `memorias_guardadas`: resultado de la extracción posterior;
- `informes`: informes producidos por especialistas en el turno;
- `tarea`: instrucción actual asignada por el supervisor;
- `delegaciones`: cantidad de especialistas ejecutados.

La función `add_messages` combina los mensajes nuevos con los mensajes
persistidos por el checkpointer.

### 3.3. Especialistas y tools

Cada instancia de `Subagente` se crea con:

- un nombre;
- una descripción de dominio;
- un prompt especializado;
- una lista de tools;
- un tipo de fuente (`RAG`, `propia` o `MCP`).

El especialista es un agente ReAct interno: puede decidir qué tool llamar y
puede llamar más de una tool para resolver su tarea. Sin embargo, hacia el
grafo devuelve un único **informe**. Esto permite que el supervisor razone
sobre resultados homogéneos, en lugar de manipular directamente todos los
detalles internos de cada tool.

### 3.4. Síntesis final

El nodo `respuesta_final` no tiene tools. Recibe únicamente:

1. los informes del turno;
2. las memorias relevantes;
3. la ventana reciente del historial.

Su función es sintetizar, no investigar. Esta separación reduce el riesgo de
que el modelo agregue datos que ninguna fuente confirmó.

## 4. Flujo completo de una consulta

El flujo del grafo es:

```text
START
  |
  v
cargar_memoria
  |
  v
supervisor <-----------------------------+
  |                                      |
  +--> conocimiento ---------------------+
  +--> datos_externos -------------------+
  +--> github ---------------------------+
  |                                      |
  +--> respuesta_final                   |
             |                           |
             v                           |
       extraer_memoria                   |
             |                           |
             v                           |
            END                          |
```

### Paso 1: recuperación de memoria larga

`cargar_memoria` toma la última consulta, la convierte en embedding y busca
hasta cuatro recuerdos relevantes del usuario en Pinecone. La búsqueda filtra
por `user_id`, por lo que un usuario nunca recibe accidentalmente recuerdos de
otro.

También reinicia los acumuladores propios del turno:

- elimina los informes de la consulta anterior;
- pone `delegaciones` en cero;
- vacía `memorias_guardadas`.

### Paso 2: decisión del supervisor

El supervisor utiliza salida estructurada con el modelo `Ruta`, que contiene:

- `siguiente`: nombre de un especialista o `FINISH`;
- `tarea`: instrucción autocontenida;
- `motivo`: justificación breve.

El supervisor no responde al usuario, no llama tools y no aporta conocimiento
propio. Su responsabilidad es seleccionar el próximo paso.

La tarea que delega debe resolver referencias. Por ejemplo, en vez de enviar
“buscá repositorios sobre esa ciudad”, debe enviar “buscá repositorios sobre
Montevideo”. Esto evita que el especialista dependa de un historial que no
recibe completo.

### Paso 3: ejecución del especialista

El subagente recibe:

- la tarea asignada;
- los recuerdos relevantes, si existen;
- los informes previos del mismo turno, si existen.

No recibe todo el historial. Esta decisión limita la contaminación entre
dominios y obliga a que la información relevante sea transmitida de manera
explícita por el supervisor.

Después de ejecutar, `_auditar_tools` recorre los mensajes internos y registra:

- qué tools fueron llamadas;
- qué tipo de tool se utilizó;
- qué devolvió cada tool;
- si al menos una tool produjo datos válidos.

El campo `con_datos` no se basa en lo que afirma el LLM, sino en el resultado
real de las tools. Esa señal se usa para controlar la incertidumbre.

### Paso 4: retorno al supervisor

El informe se agrega a `state["informes"]` y el flujo vuelve al supervisor.
Esto permite resolver consultas compuestas de manera secuencial.

Ejemplo:

```text
Consulta: "¿Cuál es la capital de Uruguay y qué repositorios de GitHub
hay relacionados con esa ciudad?"

1. supervisor -> datos_externos
2. consultar_pais("Uruguay") -> Montevideo
3. supervisor incorpora el informe
4. supervisor -> github con la tarea sobre "Montevideo"
5. búsqueda MCP
6. supervisor -> FINISH
```

El límite `MAX_DELEGACIONES = 4` evita ciclos costosos. Además, cada
especialista puede actuar como máximo una vez por consulta. Si el supervisor
intenta repetirlo, el sistema termina con los datos ya disponibles.

### Paso 5: síntesis de la respuesta

Si hubo delegaciones y todas devolvieron vacío o error, el sistema no vuelve a
consultar al LLM para “adivinar”: responde una frase de incertidumbre y enumera
los dominios que sí puede atender.

Si hay datos, `respuesta_final` genera una respuesta breve en español. Sus
instrucciones exigen:

- contestar todas las partes de la consulta;
- incluir información de todos los informes pertinentes;
- usar historial o memoria cuando correspondan;
- no usar conocimiento externo a las fuentes recibidas;
- no revelar detalles internos de la arquitectura al usuario final.

### Paso 6: extracción de memoria

Después de responder, `extraer_memoria` analiza solamente el último mensaje
humano. Es un paso separado de la generación de la respuesta.

El modelo extrae hechos estables, por ejemplo:

```text
"Soy vegetariano y me encanta la ciencia ficción."
-->
"Al usuario le gusta la ciencia ficción y es vegetariano."
```

No se guardan preguntas, pedidos puntuales ni inferencias. Cada hecho se
compara semánticamente con el recuerdo más cercano:

- `DUPLICATE`: se omite;
- `CONTRADICTION`: se reemplaza el anterior;
- `UPDATE`: se reemplaza por una versión más completa;
- `NEW`: se agrega como recuerdo nuevo.

## 5. Herramientas propias y APIs externas

[`tools_propias.py`](./tools_propias.py) define dos tools con el decorador
`@tool`.

### 5.1. `consultar_clima`

Implementa un flujo de dos llamadas:

1. geocodificación de la ciudad en Open-Meteo;
2. consulta del pronóstico usando latitud y longitud.

La tool normaliza los códigos meteorológicos WMO a texto en español y devuelve
temperatura, condición, viento, máxima, mínima y precipitación.

Si no se encuentra la ciudad o falla una API, devuelve explícitamente
`SIN_RESULTADOS` o `ERROR`.

### 5.2. `consultar_pais`

Consulta la API local de países de Open Session 1. Codifica el nombre del país
en la URL y normaliza la respuesta a campos legibles: capital, región,
población, superficie, idiomas, moneda, zonas horarias y gentilicio.

El prompt del subagente aclara que la API espera nombres en inglés. También
indica el encadenamiento correcto: primero obtener la capital y luego consultar
su clima.

## 6. RAG del subagente de conocimiento

### 6.1. Ingesta

[`ingesta_rag.py`](./ingesta_rag.py) lee el JSON y crea un vector por documento.
Para cada documento combina:

- título;
- metadatos;
- texto descriptivo.

La inclusión de metadatos permite recuperar documentos cuando la consulta
menciona un año, director, género o ingrediente, aunque esa palabra no esté
en la sinopsis.

Se usa `paraphrase-multilingual-MiniLM-L12-v2`, apropiado para consultas en
español. Los embeddings tienen dimensión 384 y se almacenan en el índice
`os3-conocimiento`.

Hay un namespace por dominio:

```text
peliculas
libros
recetas
```

Los IDs se derivan del dominio y título, por lo que la ingesta es idempotente:
volver a ejecutarla actualiza los vectores en lugar de duplicarlos.

### 6.2. Recuperación

[`rag_tools.py`](./rag_tools.py) genera tres tools:

- `buscar_peliculas`;
- `buscar_libros`;
- `buscar_recetas`.

Cada una llama a `buscar_en_indice`, consulta los cuatro vecinos más cercanos
(`TOP_K = 4`) y descarta resultados con similitud menor que `0.2`.

El resultado incluye título, metadatos, contenido y score. El prompt ordena al
subagente usar exclusivamente esos datos. Si no hay coincidencias suficientes,
la tool devuelve `SIN_RESULTADOS`.

El uso del mismo modelo de embeddings en ingesta y consulta es indispensable:
ambos textos deben pertenecer al mismo espacio vectorial para que la similitud
coseno sea significativa.

## 7. Integración MCP con GitHub

[`mcp_github.py`](./mcp_github.py) usa
`MultiServerMCPClient` y transporte `streamable_http` para descubrir tools de
un servidor MCP remoto de GitHub.

El cliente no implementa la lógica de GitHub: negocia capacidades, obtiene las
tools y las filtra. La whitelist habilita solamente operaciones de lectura:

- búsqueda de repositorios;
- lectura de archivos;
- búsqueda de código;
- búsqueda y listado de issues;
- búsqueda de usuarios.

La restricción de lectura se refuerza en dos niveles:

1. whitelist local de tools permitidas;
2. encabezado de modo read-only enviado al servidor.

Si MCP no está disponible, [`main.py`](./main.py) captura el fallo de conexión,
muestra el error y construye el resto del sistema con `tools_github = []`.
Así RAG, memoria y APIs propias continúan funcionando. El especialista de
GitHub queda explícitamente sin tools y produce un error auditable si se lo
elige.

## 8. Las dos memorias

### 8.1. Memoria de corto plazo

La memoria conversacional es el historial de mensajes mantenido por
`InMemorySaver`. El `thread_id` es la clave de recuperación.

Se utiliza para resolver referencias como:

```text
Usuario: "¿Cuál es la moneda de Brasil?"
Usuario: "¿Y cómo está el clima en Madrid?"
Usuario: "¿Cuál era el país por el que te pregunté primero?"
```

La ventana `VENTANA_HISTORIAL = 12` limita cuánto historial se reenvía a los
LLM de decisión y síntesis.

Esta memoria es de proceso: el `InMemorySaver` no constituye almacenamiento
durable entre reinicios.

### 8.2. Memoria semántica de largo plazo

[`memoria.py`](./memoria.py) usa otro índice Pinecone, `semantic-memory`, en el
namespace `open-session-3`. Se mantiene separado del índice de conocimiento
para no confundir preferencias personales con documentos del catálogo.

Cada vector contiene:

- embedding del hecho;
- `user_id`;
- texto normalizado del recuerdo.

El filtro por `user_id` es la partición lógica de la memoria. El ID combina el
usuario con un hash del texto, lo que hace que el `upsert` sea idempotente.

La CLI ofrece dos operaciones administrativas:

- `memorias`: lista todos los recuerdos del usuario;
- `olvidar`: borra todos los recuerdos de ese usuario.

## 9. Manejo de incertidumbre y grounding

El sistema aplica varias barreras contra alucinaciones:

1. los especialistas tienen la regla “usar únicamente lo que devuelvan sus
   tools”;
2. las tools devuelven marcadores explícitos de error o ausencia;
3. RAG aplica un umbral mínimo de similitud;
4. `_auditar_tools` determina objetivamente si hubo datos;
5. si ningún especialista obtuvo datos, se evita una nueva generación libre;
6. el sintetizador recibe fuentes delimitadas y no tiene tools;
7. las respuestas fuera de alcance terminan sin delegar a un especialista
   inadecuado.

Por lo tanto, “no sé” es un resultado válido del sistema, no una falla de
presentación.

## 10. Casos de prueba y qué verifican

[`casos_prueba.py`](./casos_prueba.py) organiza pruebas por grupos:

### `simples`

- RAG: recomendación de película espacial.
- tool propia: clima de Montevideo.
- MCP: estrellas de un repositorio de GitHub.

Verifican que cada dominio sea derivado al especialista correcto.

### `combinada`

Consulta la capital de Uruguay y luego busca repositorios relacionados con esa
ciudad. Verifica delegación secuencial y transferencia de resultados entre
especialistas.

### `corto`

Usa tres turnos en el mismo thread. Verifica que el sistema recuerde el país
mencionado anteriormente mediante el checkpointer.

### `largo`

Ejecuta dos procesos Python independientes:

1. el usuario declara una preferencia y una restricción alimentaria;
2. en otro proceso pide una receta.

El segundo proceso no puede depender del checkpointer del primero, por lo que
la respuesta correcta demuestra que la memoria semántica persistente funciona.

### `fuera`

Pregunta por un dato deportivo que no pertenece a ningún especialista.
Verifica que el supervisor no fuerce una delegación incorrecta y que se
reconozca la falta de cobertura.

## 11. Ejemplo de traza conceptual

Para la consulta:

```text
"Soy vegetariano. ¿Qué receta me recomendás para la cena?"
```

el flujo esperado es:

1. `cargar_memoria` recupera, si existe, la preferencia vegetariana.
2. `supervisor` elige `conocimiento`.
3. el subagente concatena la tarea con la preferencia y llama
   `buscar_recetas`.
4. Pinecone recupera documentos del namespace `recetas`.
5. el especialista informa las recetas compatibles.
6. `supervisor` elige `FINISH`.
7. `respuesta_final` sintetiza la recomendación.
8. `extraer_memoria` decide si “vegetariano” debe guardarse o actualizarse.

Para una consulta sobre GitHub, el recorrido cambia solamente en el especialista
y la fuente: el subagente de integraciones ejecuta una tool MCP de lectura.

## 12. Observaciones de diseño para el examen

### ¿Por qué el supervisor no tiene tools?

Porque su responsabilidad es de **routing**, no de resolución. Si también
investigara, se perdería la separación de responsabilidades y sería difícil
auditar de dónde salió cada dato.

### ¿Por qué cada subagente recibe una tarea autocontenida?

Para reducir el acoplamiento con el historial y evitar contaminación entre
dominios. El supervisor convierte la intención general en una tarea concreta.

### ¿Por qué hay un índice vectorial separado para memoria?

Porque el catálogo y los hechos del usuario son tipos de conocimiento
diferentes. Mezclarlos permitiría que una preferencia personal apareciera como
si fuera un documento del catálogo.

### ¿Por qué la memoria larga se recupera antes de delegar?

Porque puede modificar la tarea: una preferencia vegetariana debe llegar a la
búsqueda de recetas; una ciudad recordada puede completar una consulta
posterior sobre clima.

### ¿Por qué la extracción ocurre después de responder?

La generación de respuesta y la persistencia son responsabilidades distintas.
Además, el último mensaje humano está inequívocamente disponible y la
persistencia no bloquea la decisión de respuesta.

### ¿Qué diferencia hay entre RAG y memoria?

- **RAG** recupera conocimiento externo o documental para responder una
  consulta puntual.
- **Memoria** recupera hechos sobre el usuario para personalizar futuras
  conversaciones.

Ambos usan embeddings y Pinecone, pero tienen índices, namespaces, filtros y
propósitos distintos.

### ¿Qué diferencia hay entre una tool propia y una tool MCP?

Una tool propia implementa localmente la lógica de acceso a una API. Una tool
MCP es descubierta dinámicamente desde un servidor que expone capacidades por un
protocolo estándar. En esta solución, el código local integra y filtra las
tools de GitHub, pero no implementa la búsqueda de GitHub.

## 13. Nota sobre la trazabilidad visible

La solución imprime una traza de auditoría durante la ejecución: memoria
recuperada, decisión del supervisor, tools llamadas, resultados, informes y
memoria guardada. Esto cumple el objetivo didáctico de hacer visible el
recorrido.

En el código entregado, la ejecución principal usa `graph.ainvoke` y `print`
para mostrar esa trazabilidad. No se observa una llamada explícita a
`graph.astream`; por eso, si la evaluación exige literalmente la API de stream
de LangGraph y no solamente registros visibles, sería necesario adaptar
[`main.py`](./main.py) para consumir `graph.astream` o `graph.astream_events`.

## 14. Resumen final

La solución es un pipeline de decisión, ejecución, síntesis y aprendizaje:

```text
consulta
  -> recuperar memoria del usuario
  -> supervisor decide
  -> especialista usa sus tools
  -> informe auditable
  -> supervisor decide si falta otra parte
  -> síntesis grounded
  -> extracción y persistencia de nuevos recuerdos
```

La idea central es separar tres problemas que en un agente monolítico quedarían
mezclados:

1. **qué dominio debe actuar**: supervisor;
2. **cómo obtener el dato**: especialista y tools;
3. **cómo redactar y recordar**: síntesis final y memoria.

Esa separación proporciona modularidad, trazabilidad, control de incertidumbre y
una base clara para agregar nuevos dominios sin modificar la lógica interna de
los existentes.
