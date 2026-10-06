User: Explicar el flujo de estos tres scripts en su asociación con las arquitecturas Supervisor, Router y Swarm. La idea es que sea para un entendimiento académico, así poder documentar.

Copilot: Voy a leer los tres scripts en paralelo y relacionar cada flujo con la arquitectura que realmente implementa, distinguiendo coordinación, enrutamiento, paralelismo y handoffs.

## Visión general

Los tres scripts implementan tres formas distintas de coordinar agentes dentro de un grafo de LangGraph:

| Arquitectura | Quién decide el siguiente paso | Ejecución | Control |
|---|---|---|---|
| **Supervisor** | Un agente coordinador central | Generalmente secuencial | Centralizado |
| **Router** | Un clasificador inicial | Puede ser paralela | Predeterminada después del ruteo |
| **Swarm** | Los propios agentes | Secuencial, mediante handoffs | Descentralizado |

Una forma sencilla de diferenciarlos es:

```text
Supervisor:  Supervisor -> Worker -> Supervisor -> Worker -> ...
Router:      Router -> Worker A
                    -> Worker B   -> Synthesis
                    -> Worker C
Swarm:       Agent A <-> Agent B
```

---

# 1. Arquitectura Supervisor

Archivo: [demo6-supervisor.py]()

## Idea conceptual

En esta arquitectura existe un agente especial que cumple el papel de **coordinador central**. Los demás agentes son trabajadores especializados.

En el script participan:

- `supervisor`: decide quién debe actuar.
- `researcher`: busca información en la base de conocimiento.
- `writer`: redacta la respuesta final.

La definición de los agentes aparece en [researcher_agent y writer_agent]().

El supervisor no realiza directamente la investigación ni la redacción. Su responsabilidad es interpretar el estado actual y decidir el próximo nodo.

## Flujo

```text
START
  |
  v
Supervisor
  |
  +--> Researcher
  |       |
  |       v
  |    Supervisor
  |
  +--> Writer
          |
          v
       Supervisor
          |
          v
        FINISH
```

### Paso 1: llega la consulta

La función `run()` crea un estado inicial con un mensaje humano:

```python
{"messages": [HumanMessage(content=user_input)]}
```

Ese estado entra en el grafo desde `START`.

El primer nodo ejecutado es `supervisor`, debido a:

```python
builder.add_edge(START, "supervisor")
```

Esto se encuentra en [la construcción del grafo]().

### Paso 2: el supervisor analiza el estado

El supervisor utiliza una salida estructurada basada en el modelo `Route`:

```python
class Route(BaseModel):
    next: Literal["researcher", "writer", "FINISH"]
    reason: str
```

El uso de `Literal` limita las decisiones posibles a tres valores válidos:

- `researcher`
- `writer`
- `FINISH`

La llamada:

```python
decision = supervisor_llm.invoke(...)
```

produce una decisión estructurada.

La lógica principal está en [supervisor]().

### Paso 3: se ejecuta el investigador

En una consulta nueva, el supervisor normalmente envía el trabajo al nodo `researcher`.

El agente investigador dispone de la herramienta:

```python
search_kb
```

Su responsabilidad es recopilar hechos y devolverlos como viñetas. No debe producir el artículo final.

Cuando termina, el resultado se agrega al estado global como un mensaje identificado con el nombre `researcher`:

```python
AIMessage(content=content, name="researcher")
```

Luego, el nodo investigador devuelve el control al supervisor:

```python
goto="supervisor"
```

Esto es importante: el investigador **no decide directamente** si ahora debe escribir el autor. Siempre vuelve al coordinador.

### Paso 4: el supervisor envía el trabajo al escritor

Al recibir los hechos del investigador, el supervisor vuelve a evaluar el estado. Como ya existen datos, decide enviar la ejecución a `writer`.

El escritor recibe la conversación completa, incluidos los hechos aportados por `researcher`, y redacta un párrafo final.

El escritor tampoco decide por sí mismo qué agente debe actuar después. Al terminar, devuelve nuevamente el control al supervisor.

### Paso 5: finalización

Cuando el supervisor detecta que el escritor ya produjo la respuesta final, devuelve:

```python
Command(goto=END)
```

El grafo termina y `run()` toma el último mensaje como respuesta:

```python
result['messages'][-1].content
```

## Característica fundamental

El supervisor conserva siempre el control:

```text
researcher -> supervisor
writer     -> supervisor
```

Los trabajadores no se llaman entre sí. La relación es centralizada:

```text
         Supervisor
        /          \
Researcher      Writer
```

## Ventajas académicas

- Permite aplicar políticas centrales.
- Facilita registrar y auditar cada decisión.
- Es sencillo agregar más especialistas.
- El supervisor puede decidir repetir una etapa, cambiar de especialista o finalizar.
- Es adecuado cuando el flujo depende del estado producido por pasos anteriores.

## Limitaciones

- El supervisor puede convertirse en un cuello de botella.
- Cada transición implica otra decisión del coordinador.
- La calidad general depende mucho del prompt y del razonamiento del supervisor.
- Los trabajadores tienen menor autonomía.

## Resumen académico

> La arquitectura Supervisor implementa una coordinación jerárquica. Un agente coordinador mantiene la visión global del estado y dirige a agentes especialistas, que ejecutan tareas acotadas y devuelven el control al coordinador después de cada intervención.

---

# 2. Arquitectura Router con ejecución paralela

Archivo: [demo7-router-parallel.py]()

## Idea conceptual

En esta arquitectura existe un nodo inicial llamado `classify` que funciona como **router**.

Su objetivo es determinar qué especialistas deben participar en la respuesta.

A diferencia del supervisor, el router no supervisa continuamente cada agente. Toma una decisión inicial y genera las ejecuciones necesarias.

Los especialistas disponibles son:

- `billing`
- `technical`
- `legal`

La clasificación se representa mediante:

```python
class Classification(BaseModel):
    specialists: List[Literal["billing", "technical", "legal"]]
```

La definición está en [Classification]().

## Flujo

```text
START
  |
  v
Classify / Router
  |
  +----> Billing specialist ----+
  |                              |
  +----> Technical specialist ---+--> Synthesize --> END
  |                              |
  +----> Legal specialist -------+
```

### Paso 1: llega la consulta

El estado inicial contiene:

```python
{
    "question": user_input,
    "specialists": [],
    "answers": [],
    "final": ""
}
```

El flujo comienza en `classify`:

```python
builder.add_edge(START, "classify")
```

### Paso 2: el router clasifica la consulta

La función `classify()` utiliza el modelo con salida estructurada:

```python
decision = llm.with_structured_output(Classification).invoke(...)
```

El modelo determina todos los dominios involucrados.

Por ejemplo, ante:

```text
Me cobraron dos veces, la aplicación falla y quiero eliminar mis datos.
```

el router podría producir:

```python
["billing", "technical", "legal"]
```

La salida se almacena en:

```python
"specialists": decision.specialists
```

A diferencia del supervisor, esta clasificación no tiene necesariamente que elegir un único agente. Puede seleccionar varios.

## Paso 3: fan-out

La función `fan_out()` transforma la lista de especialistas en varias ejecuciones:

```python
return [
    Send("specialist", {
        "question": state["question"],
        "specialist": name
    })
    for name in state["specialists"]
]
```

Conceptualmente, `Send` crea una ejecución independiente del nodo `specialist` para cada dominio.

Por ejemplo:

```text
Send("specialist", {"specialist": "billing"})
Send("specialist", {"specialist": "technical"})
Send("specialist", {"specialist": "legal"})
```

Esto constituye un patrón **fan-out**:

```text
                 +--> Billing
Router ----------+--> Technical
                 +--> Legal
```

## Paso 4: trabajo paralelo

Cada ejecución recibe un `WorkerInput` independiente:

```python
class WorkerInput(TypedDict):
    question: str
    specialist: str
```

El nodo `specialist()` identifica el dominio:

```python
name = state["specialist"]
```

y aplica el prompt correspondiente:

```python
SPECIALISTS[name]
```

Cada especialista produce una respuesta parcial.

Por ejemplo:

```text
billing: explicación del cobro duplicado
technical: explicación del fallo de la aplicación
legal: información sobre eliminación de datos
```

## Paso 5: reducción de resultados

Cada especialista devuelve:

```python
{"answers": [f"{name}: {response.content}"]}
```

El campo `answers` está definido como:

```python
answers: Annotated[list, operator.add]
```

Esto configura un **reducer aditivo**. En vez de que cada ejecución sobrescriba las respuestas anteriores, las respuestas se acumulan:

```text
answers iniciales: []

billing    -> ["billing: ..."]
technical  -> ["billing: ...", "technical: ..."]
legal      -> ["billing: ...", "technical: ...", "legal: ..."]
```

Este mecanismo es necesario porque varias ramas escriben sobre el mismo campo.

## Paso 6: síntesis

Una vez completadas las ejecuciones paralelas, el grafo llega a `synthesize`.

La función concatena las respuestas:

```python
joined = "\n".join(state["answers"])
```

y realiza una nueva llamada al modelo para fusionarlas:

```python
response = llm.invoke(...)
```

El resultado se almacena en:

```python
"final": response.content
```

Finalmente, el grafo termina:

```python
builder.add_edge("synthesize", END)
```

## Característica fundamental

El router toma una decisión de selección al principio, pero no gobierna individualmente cada paso posterior:

```text
Router -> seleccionar especialistas
Especialistas -> trabajar independientemente
Synthesize -> combinar resultados
```

## Ventajas académicas

- Permite resolver consultas multidominio.
- Reduce el tiempo total cuando las tareas son independientes.
- Es una buena representación del patrón map-reduce:
  - **Map**: cada especialista produce una respuesta.
  - **Reduce**: `synthesize` combina los resultados.
- El número de especialistas participantes puede variar por consulta.
- El flujo es relativamente fácil de visualizar.

## Limitaciones

- Los especialistas no comparten resultados entre sí durante su ejecución.
- Si un especialista necesita la respuesta de otro, este patrón no es suficiente.
- La síntesis puede perder matices o introducir inconsistencias.
- El router decide al principio y no reevalúa la situación después de cada respuesta.
- Si no se selecciona ningún especialista, el flujo puede quedar sin resultados útiles para sintetizar.

## Resumen académico

> La arquitectura Router separa la clasificación de la ejecución. Un router analiza la consulta, selecciona uno o varios especialistas y distribuye el trabajo. Las respuestas parciales se combinan posteriormente mediante una etapa de síntesis. Cuando las tareas son independientes, la ejecución puede realizarse en paralelo.

---

# 3. Arquitectura Swarm mediante handoffs

Archivo: [demo8-swarm-handoffs.py]()

## Idea conceptual

En un **Swarm** no existe un supervisor central que decida continuamente qué agente debe actuar.

Los agentes pueden transferirse el control directamente mediante herramientas de handoff.

En este ejemplo participan:

- `researcher`
- `writer`

Cada agente conoce al otro y tiene una herramienta para entregarle el control:

```python
to_writer = create_handoff_tool(...)
to_researcher = create_handoff_tool(...)
```

La fábrica de herramientas está definida en [create_handoff_tool]().

## Flujo

```text
START
  |
  v
Entry
  |
  v
Researcher
  |
  | transfer_to_writer
  v
Writer
  |
  | puede finalizar
  | o transfer_to_researcher
  v
Researcher
```

El flujo no tiene un nodo supervisor que reciba todas las decisiones.

## Paso 1: entrada

El grafo comienza en `entry`:

```python
builder.add_edge(START, "entry")
```

La función `entry()` consulta el agente activo:

```python
target = state.get("active_agent") or "researcher"
```

Si es la primera interacción, no existe un agente activo y se comienza con `researcher`.

Si ya existe un checkpoint para el mismo `thread_id`, se retoma el agente que tenía el control:

```python
active_agent
```

Esto permite conservar la continuidad de la conversación.

## Paso 2: el investigador trabaja

El investigador tiene dos herramientas:

```python
tools=[search_kb, to_writer]
```

Puede:

1. Consultar la base de conocimiento.
2. Transferir el control al escritor.

Su prompt le indica:

```text
As soon as you have the facts, call transfer_to_writer.
Never write the final text yourself.
```

Por lo tanto, el investigador reúne la información y luego ejecuta la herramienta `transfer_to_writer`.

## Paso 3: se ejecuta el handoff

La herramienta de handoff devuelve:

```python
Command(
    goto=agent_name,
    graph=Command.PARENT,
    update={
        "messages": state["messages"] + [tool_message],
        "active_agent": agent_name
    },
)
```

Esta instrucción hace tres cosas:

1. Cambia el destino de la ejecución:
   ```python
   goto=agent_name
   ```

2. Indica que la navegación debe regresar al grafo padre:
   ```python
   graph=Command.PARENT
   ```

3. Actualiza el estado:
   ```python
   "active_agent": agent_name
   ```

Por ejemplo:

```text
researcher
    |
    | transfer_to_writer
    v
writer
```

El mensaje de herramienta:

```python
"Control transferred to writer."
```

también queda incorporado a la conversación.

## Paso 4: el escritor toma el control

El escritor recibe todo el historial, incluidos los hechos reunidos por el investigador.

Tiene una sola herramienta de coordinación:

```python
tools=[to_researcher]
```

Si los hechos son suficientes, redacta la respuesta final.

Si no son suficientes, puede devolver el control al investigador:

```text
writer -> researcher
```

Por ejemplo:

```text
El escritor detecta que faltan datos
        |
        v
transfer_to_researcher
        |
        v
El investigador continúa trabajando
```

## Paso 5: finalización o nuevo handoff

El escritor puede:

- Terminar y producir una respuesta.
- Transferir el control nuevamente al investigador.

Esto genera una red descentralizada:

```text
Researcher <------> Writer
```

No hay una entidad central equivalente al supervisor.

## Papel de `active_agent`

El campo:

```python
class SwarmState(MessagesState):
    active_agent: str
```

cumple una función de control.

El campo `messages` contiene el contenido conversacional, mientras que `active_agent` indica quién debe recibir el próximo turno.

Es importante distinguir:

```text
messages      = memoria de contenido
active_agent  = memoria de control
```

El `InMemorySaver` permite preservar este estado para el mismo `thread_id` durante la ejecución del proceso.

## Ventajas académicas

- Los agentes son más autónomos.
- La colaboración se expresa de forma natural mediante handoffs.
- No se necesita un coordinador central.
- Es apropiado cuando los agentes deben negociar o reaccionar dinámicamente.
- El agente que tiene el contexto puede decidir quién debe continuar.

## Limitaciones

- El control del flujo es menos predecible.
- Pueden producirse ciclos entre agentes.
- Es más difícil auditar quién tomó cada decisión.
- La calidad depende de que los prompts definan correctamente cuándo transferir el control.
- Es necesario limitar la recursión, como se hace con:
  ```python
  "recursion_limit": 15
  ```

## Resumen académico

> La arquitectura Swarm implementa una coordinación descentralizada. Cada agente puede transferir explícitamente el control a otro agente mediante una herramienta de handoff. La decisión de continuar no pertenece a un supervisor, sino al agente que está activo.

---

# Comparación detallada del control

## Supervisor

```text
             decide
              |
              v
        +-------------+
        | Supervisor  |
        +-------------+
          |         |
          v         v
     Researcher  Writer
          |         |
          +----+----+
               |
               v
          Supervisor
```

El supervisor funciona como una autoridad central.

Los trabajadores no conocen necesariamente el flujo completo.

## Router

```text
             clasifica
                |
                v
          +-----------+
          |  Router   |
          +-----------+
          /     |      \
         v      v       v
    Billing Technical Legal
         \      |       /
          \     |      /
             v  v  v
          Synthesize
```

El router decide la distribución inicial. Después, las ramas trabajan de forma independiente y sus resultados se reducen.

## Swarm

```text
       +------------+
       | Researcher |
       +------------+
          |      ^
          |      |
          v      |
       +------------+
       |   Writer   |
       +------------+
```

El control circula entre agentes mediante handoffs.

---

# Diferencia entre `Command` y `Send`

Los scripts muestran dos mecanismos importantes de LangGraph.

## `Command`

Se utiliza principalmente para dirigir una ejecución a un destino específico y, opcionalmente, actualizar el estado.

En el supervisor:

```python
Command(goto=decision.next)
```

Esto significa:

> “El próximo nodo que debe ejecutarse es el que decidió el supervisor.”

En el Swarm:

```python
Command(
    goto=agent_name,
    graph=Command.PARENT,
    update=...
)
```

Esto significa:

> “Transfiere el control a otro agente y actualiza el estado del grafo padre.”

## `Send`

Se utiliza para crear ejecuciones dinámicas, normalmente en un patrón de fan-out.

En el router:

```python
Send("specialist", {...})
```

Esto significa:

> “Ejecuta el nodo `specialist` con este estado parcial.”

Si hay tres especialistas, se generan tres ejecuciones independientes.

---

# Diferencia en el uso del estado

## Supervisor

Usa principalmente:

```python
MessagesState
```

El historial de mensajes es la memoria compartida. Los resultados del investigador y del escritor se agregan como mensajes con nombre.

```text
messages:
  HumanMessage
  AIMessage(name="researcher")
  AIMessage(name="writer")
```

## Router

Usa un estado tipado explícito:

```python
class RouterState(TypedDict):
    question: str
    specialists: list
    answers: Annotated[list, operator.add]
    final: str
```

La información está separada por función:

- `question`: consulta original.
- `specialists`: dominios seleccionados.
- `answers`: resultados parciales.
- `final`: respuesta sintetizada.

## Swarm

Combina:

```python
MessagesState
active_agent: str
```

Por eso mantiene dos tipos de información:

```text
Conversación: messages
Control:      active_agent
```

---

# Comparación del ciclo de vida de una consulta

## Supervisor

```text
1. Recibir consulta
2. Consultar supervisor
3. Ejecutar investigador
4. Volver al supervisor
5. Ejecutar escritor
6. Volver al supervisor
7. Finalizar
```

## Router

```text
1. Recibir consulta
2. Clasificar dominios
3. Crear ramas de especialistas
4. Ejecutar ramas en paralelo
5. Acumular respuestas
6. Sintetizar
7. Finalizar
```

## Swarm

```text
1. Recibir consulta
2. Determinar agente activo
3. Ejecutar agente
4. El agente decide si hace handoff
5. Otro agente continúa
6. Repetir o finalizar
```

---

# Relación con los patrones de diseño

## Supervisor como coordinación jerárquica

Es comparable a una estructura organizacional:

```text
Gerente
  |
  +-- Investigador
  +-- Redactor
```

El gerente asigna trabajo y revisa el avance.

## Router como distribución de tareas

Es comparable a una central de clasificación:

```text
Solicitud
   |
   +-- Área de facturación
   +-- Área técnica
   +-- Área legal
```

Cada área trabaja sobre su parte y luego una etapa final consolida la respuesta.

## Swarm como equipo colaborativo

Es comparable a un equipo donde los miembros se pasan el trabajo directamente:

```text
Investigador: "Ya reuní los datos; continúa el redactor."
Redactor: "Faltan datos; vuelve a investigar."
```

No existe un gerente que intervenga en cada transición.

---

# ¿Cuándo usar cada arquitectura?

| Necesidad | Arquitectura recomendada |
|---|---|
| Se necesita control central y trazabilidad | Supervisor |
| Hay tareas independientes que pueden correr simultáneamente | Router |
| La consulta combina varios dominios | Router |
| Los agentes deben colaborar de forma dinámica | Swarm |
| El orden de trabajo es conocido y secuencial | Supervisor |
| Se desea que los agentes decidan entre ellos | Swarm |
| Se necesita aplicar reglas globales en cada paso | Supervisor |
| Se busca reducir la latencia de tareas independientes | Router paralelo |

---

# Conclusión

Los tres scripts no representan simplemente tres maneras de llamar a un modelo. Representan tres modelos diferentes de **control de flujo multi-agente**:

1. **Supervisor**: un coordinador central decide y recupera el control después de cada trabajador.
2. **Router**: un clasificador inicial distribuye la consulta entre especialistas, que pueden ejecutarse en paralelo, y una etapa de síntesis combina los resultados.
3. **Swarm**: los agentes colaboran de forma descentralizada y se transfieren el control mediante handoffs.

La diferencia principal puede resumirse así:

```text
Supervisor = coordinación centralizada
Router     = distribución y agregación
Swarm      = delegación descentralizada
```

En términos de grafos:

```text
Supervisor:
  control central con retornos al coordinador

Router:
  fan-out / fan-in

Swarm:
  navegación entre agentes mediante handoffs
```