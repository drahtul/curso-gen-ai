"""Sistema multiagente: supervisor + subagentes especialistas (LangGraph).

- supervisor: solo decide (with_structured_output) a quién delegar y con qué
  tarea, o si ya se puede terminar. No tiene tools ni redacta contenido.
- subagentes: cada uno es un agente ReAct con SUS tools; reciben una tarea
  autocontenida escrita por el supervisor y devuelven un informe.
- respuesta_final: arma la respuesta al usuario usando únicamente los informes
  del turno, el historial y las memorias. Si no alcanza, lo dice.
- cargar_memoria / extraer_memoria: memoria semántica de largo plazo (Pinecone).
- El historial de la conversación vive en el checkpointer (memoria de corto
  plazo, por thread_id).
"""
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from memoria import extraer_y_guardar, recuperar_memorias
from rag_tools import buscar_libros, buscar_peliculas, buscar_recetas
from tools_propias import consultar_clima, consultar_pais

load_dotenv()

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-nano")

# Tope de delegaciones por turno: evita ciclos de routing y limita el costo de
# inferencia si el supervisor no logra cerrar una consulta.
MAX_DELEGACIONES = 4
# Ventana de historial que ve el supervisor: suficiente contexto sin reenviar
# conversaciones completas en cada decisión.
VENTANA_HISTORIAL = 12

SIN_INFO = "No cuento con información suficiente para responder esa consulta."
SIN_INFO_COMPLETO = (
    SIN_INFO + " Puedo ayudarte con películas, libros, recetas, clima, datos de "
    "países y búsquedas en GitHub."
)

TOOLS_RAG = [buscar_peliculas, buscar_libros, buscar_recetas]
TOOLS_PROPIAS = [consultar_clima, consultar_pais]


class EstadoSistema(TypedDict):
    # add_messages combina mensajes nuevos con los guardados por thread_id.
    messages: Annotated[list, add_messages]
    user_id: str
    memorias: list[str]                       # recuperadas de largo plazo
    memorias_guardadas: list[str]
    informes: list[dict]                      # lo que reportó cada subagente en el turno
    tarea: str                                # tarea que el supervisor asignó
    delegaciones: int


# ------------------------------------------------------------- subagentes

# Estas reglas funcionan como contrato de grounding: una tool es la fuente de
# datos y el informe debe reconocer explícitamente cuando no encontró nada.
REGLAS_SUBAGENTE = """
Reglas de tu informe:
- Usá únicamente lo que devuelvan tus tools; nada de conocimiento propio.
- Devolvé un informe breve en español con los datos que encontraste. Si falta
  algún dato puntual, informá lo que sí hay y aclará qué faltó.
- Si la tarea NO es de tu dominio, no llames a ninguna tool y respondé solo:
  "Esta tarea no corresponde a mi dominio." """

PROMPT_CONOCIMIENTO = """Sos el subagente de CONOCIMIENTO de una plataforma de entretenimiento.
Resolvés consultas sobre películas, libros y recetas usando SOLO el catálogo
propio en bases vectoriales (buscar_peliculas, buscar_libros, buscar_recetas).

- Nunca respondas de memoria: buscá siempre.
- Si te pasan preferencias del usuario (ej: es vegetariano, le gusta la ciencia
  ficción), tenelas en cuenta al formular la búsqueda y al elegir resultados.
""" + REGLAS_SUBAGENTE

PROMPT_DATOS_EXTERNOS = """Sos el subagente de DATOS EXTERNOS. Resolvés clima y datos de países con
tus tools (consultar_clima, consultar_pais).

- consultar_pais espera el nombre del país en INGLÉS (Japón -> Japan).
- Para el clima de la capital de un país: primero consultar_pais, después
  consultar_clima con la capital obtenida.
- Llamá solo a las tools necesarias para lo que pide la tarea: si piden la
  capital o la moneda, no consultes el clima.
""" + REGLAS_SUBAGENTE

PROMPT_GITHUB = """Sos el subagente de INTEGRACIONES con GitHub. Resolvés consultas sobre
repositorios, código, issues y usuarios de GitHub con las tools del servidor
MCP de GitHub. Solo tenés permisos de lectura.

- Para buscar repositorios usá search_repositories con la sintaxis de búsqueda
  de GitHub (ej: 'tokyo weather', 'repo:owner/nombre', 'topic:recipes
  language:python') y perPage 5 como máximo.
- Para repos SOBRE un lugar o tema usá 'topic:<tema>' en minúsculas y en
  inglés (ej: 'topic:montevideo', 'topic:tokyo', 'topic:recipes'): una búsqueda
  de texto libre trae repos que solo nombran la palabra de pasada.
- Descartá resultados que solo coinciden en el nombre pero no tienen relación
  con lo pedido (ej: la librería de Rust 'tokio' no tiene que ver con la ciudad).
- Informá nombre completo del repo, descripción, estrellas y URL.
""" + REGLAS_SUBAGENTE


# Cómo avisa cada fuente que no encontró nada: las tools propias y de RAG
# empiezan con SIN_RESULTADOS/ERROR; el MCP de GitHub devuelve total_count 0.
MARCAS_MCP_VACIO = ('"total_count":0',)


def _es_resultado_vacio(texto: str) -> bool:
    return texto.startswith(("SIN_RESULTADOS", "ERROR")) or any(m in texto for m in MARCAS_MCP_VACIO)


class Subagente:
    def __init__(self, nombre: str, descripcion: str, tools: list, prompt: str, tipo_tools: str):
        self.nombre = nombre
        self.descripcion = descripcion
        self.tipo_tools = tipo_tools
        self.agente = create_agent(model=llm, tools=tools, system_prompt=prompt) if tools else None

    async def ejecutar(self, state: EstadoSistema) -> Command:
        tarea = state["tarea"]
        print(f"\n  [{self.nombre}] tarea: {tarea}")

        if self.agente is None:
            respuesta = f"ERROR: el subagente '{self.nombre}' no tiene tools disponibles."
            tools_usadas, con_datos = [], False
        else:
            # El subagente NO ve el historial completo: recibe una tarea
            # autocontenida y contexto mínimo. Así se reduce la contaminación
            # entre dominios y se hace explícito qué información puede usar.
            contexto = []
            if state["memorias"]:
                contexto.append("Datos conocidos del usuario:\n" + "\n".join(f"- {m}" for m in state["memorias"]))
            if state["informes"]:
                contexto.append("Resultados previos de otros especialistas en esta consulta:\n" + "\n".join(
                    f"- [{i['subagente']}] {i['respuesta']}" for i in state["informes"]))
            entrada = tarea + ("\n\n" + "\n\n".join(contexto) if contexto else "")

            # Cada especialista puede ejecutar varias tools internamente, pero
            # devuelve un único informe que el supervisor incorpora al estado.
            resultado = await self.agente.ainvoke({"messages": [HumanMessage(content=entrada)]})
            tools_usadas, con_datos = self._auditar_tools(resultado["messages"])
            respuesta = resultado["messages"][-1].content

        print(f"  [{self.nombre}] informe: {respuesta[:300]}{'...' if len(respuesta) > 300 else ''}")
        if not con_datos:
            print(f"  [{self.nombre}] sin datos: ninguna tool devolvió resultados")
        informe = {"subagente": self.nombre, "tarea": tarea, "respuesta": respuesta,
                   "tools": tools_usadas, "con_datos": con_datos}
        return Command(
            update={"informes": state["informes"] + [informe], "delegaciones": state["delegaciones"] + 1},
            goto="supervisor",
        )

    def _auditar_tools(self, mensajes: list) -> tuple[list[str], bool]:
        """Loguea las tool calls y dice si alguna devolvió datos reales.

        'con_datos' sale de lo que devolvieron las tools, no de lo que el LLM
        dice en su informe: es la señal objetiva que usa la respuesta final
        para no inventar cuando nadie encontró nada.
        """
        usadas, con_datos = [], False
        for m in mensajes:
            if isinstance(m, AIMessage):
                for llamada in m.tool_calls:
                    print(f"    [tool {self.tipo_tools}] {llamada['name']}({llamada['args']})")
                    usadas.append(f"{llamada['name']} ({self.tipo_tools})")
            elif isinstance(m, ToolMessage):
                texto = str(m.content).replace("\n", " ")
                print(f"    [resultado] {texto[:160]}{'...' if len(texto) > 160 else ''}")
                if m.status != "error" and not _es_resultado_vacio(texto):
                    con_datos = True
        return usadas, con_datos


# ------------------------------------------------------------- supervisor

class Ruta(BaseModel):
    siguiente: Literal["conocimiento", "datos_externos", "github", "FINISH"] = Field(
        description="Subagente que debe actuar ahora, o FINISH si ya se puede responder."
    )
    tarea: str = Field(
        description=(
            "Instrucción autocontenida para el subagente elegido, con todos los datos "
            "necesarios (resolviendo referencias del historial y de las memorias, y "
            "usando resultados de informes previos). Vacía si siguiente es FINISH."
        )
    )
    motivo: str = Field(description="Una oración breve que justifica la decisión.")


def _prompt_supervisor(subagentes: list[Subagente]) -> str:
    equipo = "\n".join(f"- {s.nombre}: {s.descripcion}" for s in subagentes)
    return f"""Sos el SUPERVISOR de un sistema multiagente. NO respondés consultas ni
aportás datos: tu único trabajo es decidir qué especialista actúa a continuación
y con qué tarea, o terminar (FINISH).

Especialistas disponibles:
{equipo}

Cómo decidir:
1. Mirá la última consulta del usuario, el historial y los informes que ya
   llegaron en este turno.
2. Si falta información que un especialista puede obtener, delegale una tarea
   concreta y autocontenida (sin pronombres como "esa ciudad": escribí el dato).
3. Si la consulta necesita encadenar dominios (ej: obtener la capital de un
   país y después buscar en GitHub repositorios sobre esa ciudad), delegá un paso por
   vez y en la tarea siguiente incluí el dato que devolvió el informe anterior,
   copiado tal cual (si el informe dice "Tokyo", escribí "Tokyo": no lo traduzcas).
4. Vos no aportás datos: no completes una tarea con lo que "ya sabés". Si la
   consulta necesita un dato (la capital de un país, el clima, un libro),
   ese dato tiene que venir del informe de un especialista. Ejemplo:
   "repos de GitHub sobre la capital de Japón" -> primero datos_externos
   (capital de Japón), después github con la ciudad que devolvió el informe.
5. Usá las memorias del usuario para completar la tarea (ej: si sabés su
   ciudad o sus gustos, incluilos).
6. Elegí FINISH cuando:
   - los informes ya responden lo que el usuario PREGUNTÓ. No busques
     información adicional que no pidió (si preguntó el clima de Madrid, no
     hace falta buscar repos sobre Madrid); o
   - la respuesta sale directamente del historial de la conversación o de las
     memorias del usuario (ej: "¿cómo me llamo?", "¿qué ciudad te dije?"); o
   - la consulta no corresponde a ningún especialista (deportes, política,
     biografías, cálculos, etc.): no la delegues a alguien que no puede
     resolverla; o
   - un especialista ya informó que no tiene el dato (no repitas la misma tarea); o
   - el mensaje es un saludo, un agradecimiento o el usuario solo cuenta algo
     de sí mismo sin pedir nada (eso lo guarda otro paso del sistema).
7. Cada especialista actúa como máximo UNA vez por consulta: si ya aparece en
   los informes de este turno, no lo vuelvas a elegir (aunque haya informado
   que no encontró nada). En ese caso elegí otro especialista o FINISH."""


class RespuestaFinal(BaseModel):
    fuente: Literal["informes", "historial", "memoria", "charla", "ninguna"] = Field(
        description=(
            "De dónde sale la respuesta: 'informes' (datos de los especialistas), "
            "'historial' (algo dicho antes en esta conversación), 'memoria' (lo que "
            "se sabe del usuario), 'charla' (saludo, agradecimiento o el usuario "
            "contó algo de sí sin pedir datos), 'ninguna' (no hay fuente que responda)."
        )
    )
    respuesta: str = Field(description="Respuesta final al usuario, en español.")


def construir_grafo(tools_github: list):
    # Las tools se inyectan al construir el grafo: esto permite que la falta de
    # MCP deje al especialista disponible como error explícito, sin romper los
    # otros dominios.
    subagentes = [
        Subagente(
            "conocimiento",
            "RAG sobre el catálogo propio de PELÍCULAS, LIBROS y RECETAS "
            "(recomendaciones, sinopsis, directores, autores, ingredientes).",
            TOOLS_RAG, PROMPT_CONOCIMIENTO, "RAG",
        ),
        Subagente(
            "datos_externos",
            "Tools propias de CLIMA actual de una ciudad y DATOS DE PAÍSES "
            "(capital, moneda, población, idiomas). Es la ÚNICA fuente válida para "
            "datos de países.",
            TOOLS_PROPIAS, PROMPT_DATOS_EXTERNOS, "propia",
        ),
        Subagente(
            "github",
            "Servidor MCP de GitHub: buscar REPOSITORIOS, código, issues y usuarios "
            "de GitHub, o leer archivos de un repo. Es el único que accede a GitHub: "
            "elegilo siempre que la consulta mencione GitHub, repos o estrellas.",
            tools_github, PROMPT_GITHUB, "MCP",
        ),
    ]
    supervisor_llm = llm.with_structured_output(Ruta)
    respuesta_llm = llm.with_structured_output(RespuestaFinal)
    prompt_supervisor = _prompt_supervisor(subagentes)

    def cargar_memoria(state: EstadoSistema) -> dict:
        # Este nodo es el comienzo de cada turno: recupera memoria larga y
        # reinicia los acumuladores que solo pertenecen a la consulta actual.
        consulta = state["messages"][-1].content
        memorias = recuperar_memorias(state["user_id"], consulta)
        print(f"\n  [memoria] recuperadas ({len(memorias)}): {memorias or 'ninguna'}")
        # Arranca un turno nuevo: se limpian los informes del turno anterior.
        return {"memorias": memorias, "informes": [], "tarea": "", "delegaciones": 0, "memorias_guardadas": []}

    def supervisor(state: EstadoSistema) -> Command:
        # El supervisor decide rutas, pero nunca redacta la respuesta ni llama
        # tools: esa separación hace auditable la procedencia de cada dato.
        if state["delegaciones"] >= MAX_DELEGACIONES:
            print(f"\n  [supervisor] tope de {MAX_DELEGACIONES} delegaciones alcanzado -> FINISH")
            return Command(goto="respuesta_final")

        contexto = "Memorias del usuario:\n" + ("\n".join(f"- {m}" for m in state["memorias"]) or "(ninguna)")
        contexto += "\n\nInformes recibidos en este turno:\n" + ("\n".join(
            f"- [{i['subagente']}] tarea: {i['tarea']}\n  respuesta: {i['respuesta']}" for i in state["informes"]
        ) or "(ninguno todavía)")
        ya_actuaron = sorted({i["subagente"] for i in state["informes"]})
        if ya_actuaron:
            contexto += f"\n\nYa actuaron (NO elegibles otra vez): {', '.join(ya_actuaron)}"
            # Con modelos chicos ayuda cerrar con la pregunta concreta a decidir.
            contexto += (
                f"\n\nConsulta actual del usuario: {state['messages'][-1].content}\n"
                "¿Los informes de arriba ya responden TODO lo que esa consulta pregunta? "
                "Si es así, elegí FINISH. Solo delegá si falta una parte que el usuario pidió."
            )

        mensajes = (
            [SystemMessage(content=prompt_supervisor)]
            + state["messages"][-VENTANA_HISTORIAL:]
            + [SystemMessage(content=contexto)]
        )
        ruta = supervisor_llm.invoke(mensajes)
        print(f"\n  [supervisor] -> {ruta.siguiente} | motivo: {ruta.motivo}")

        if ruta.siguiente == "FINISH":
            return Command(goto="respuesta_final")
        # Resguardo anti-bucle (no es routing): cada especialista resuelve su
        # parte de una sola vez, con todas las tool calls que necesite. Si el
        # LLM insiste en volver a uno que ya informó, es que no hay más que sacar.
        if any(i["subagente"] == ruta.siguiente for i in state["informes"]):
            print(f"  [supervisor] '{ruta.siguiente}' ya informó en este turno -> FINISH")
            return Command(goto="respuesta_final")
        return Command(goto=ruta.siguiente, update={"tarea": ruta.tarea})

    def respuesta_final(state: EstadoSistema) -> dict:
        informes = state["informes"]
        # Si hubo delegaciones y TODAS volvieron sin dato, no hay nada que
        # redactar: se responde la frase de incertidumbre sin pasar por el LLM,
        # que de otro modo tiende a "ayudar" con conocimiento propio.
        if informes and not any(i["con_datos"] for i in informes):
            print("\n  [respuesta_final] ningún especialista obtuvo datos -> sin información")
            return {"messages": [AIMessage(content=SIN_INFO_COMPLETO)]}

        # La voz final recibe fuentes delimitadas y la ventana de historial,
        # pero no acceso directo a herramientas: sintetiza, no investiga.
        texto_informes = "\n\n".join(f"[{i['subagente']}]\n{i['respuesta']}" for i in informes) or "(ninguno)"
        memorias = "\n".join(f"- {m}" for m in state["memorias"]) or "(ninguna)"
        instrucciones = f"""Sos la voz de un asistente de entretenimiento e información general.
Redactá la respuesta final al usuario en español, breve y concreta.

Fuentes que PODÉS usar (y ninguna otra):
1. Informes de los especialistas en este turno:
{texto_informes}

2. Lo que sabés del usuario (memoria de largo plazo):
{memorias}

3. El historial de esta conversación.

Reglas:
- Si los informes traen datos que responden la consulta, respondé con esos
  datos (aunque algún otro informe diga que no encontró nada).
- Si la respuesta está en el historial o en lo que sabés del usuario, usala.
- Respondé TODAS las partes de la consulta (si pregunta "¿qué país te dije y
  cuál es su capital?", nombrá el país y la capital).
- Cuando hay varios informes con datos, cada uno responde una parte distinta:
  incluí el contenido de CADA uno (ej: la capital Y la lista de repositorios,
  con sus nombres). Ser breve no es omitir partes.
- Si el usuario saluda, agradece o solo cuenta algo de sí mismo, respondé
  cordialmente (podés confirmar que lo vas a tener en cuenta) sin aportar datos.
- PROHIBIDO usar conocimiento propio: aunque sepas la respuesta, si no está en
  las fuentes de arriba, la fuente es 'ninguna'.
- No menciones a los especialistas ni detalles internos del sistema."""
        final = respuesta_llm.invoke([SystemMessage(content=instrucciones)] + state["messages"][-VENTANA_HISTORIAL:])
        print(f"\n  [respuesta_final] fuente declarada: {final.fuente}")
        # Si algún especialista trajo datos, manda el contenido de los informes.
        # La fuente declarada solo decide cuando no hubo delegaciones (respuestas
        # desde el historial, la memoria o charla, o una consulta fuera de alcance).
        if not informes and final.fuente == "ninguna":
            texto = SIN_INFO_COMPLETO
        else:
            texto = final.respuesta
        return {"messages": [AIMessage(content=texto)]}

    def extraer_memoria(state: EstadoSistema) -> dict:
        # La extracción ocurre después de responder: la conversación no se
        # demora por el paso de persistencia y el último mensaje es inequívoco.
        ultimo_humano = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        guardadas = extraer_y_guardar(state["user_id"], ultimo_humano.content)
        print(f"\n  [memoria] extracción: {guardadas or 'nada para guardar'}")
        return {"memorias_guardadas": guardadas}

    # El grafo hace visible el ciclo supervisor -> especialista -> supervisor y
    # garantiza el cierre lineal respuesta -> extracción -> fin.
    builder = StateGraph(EstadoSistema)
    builder.add_node("cargar_memoria", cargar_memoria)
    builder.add_node(
        "supervisor", supervisor,
        destinations=tuple(s.nombre for s in subagentes) + ("respuesta_final",),
    )
    for s in subagentes:
        builder.add_node(s.nombre, s.ejecutar, destinations=("supervisor",))
    builder.add_node("respuesta_final", respuesta_final)
    builder.add_node("extraer_memoria", extraer_memoria)

    builder.add_edge(START, "cargar_memoria")
    builder.add_edge("cargar_memoria", "supervisor")
    builder.add_edge("respuesta_final", "extraer_memoria")
    builder.add_edge("extraer_memoria", END)

    return builder.compile(checkpointer=InMemorySaver())
