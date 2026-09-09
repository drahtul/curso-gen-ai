"""
Agente multidominio de la Open Session 2.

Arma el agente ReAct "a mano" con LangGraph, siguiendo el mismo patrón que
usa el profe en la Unidad 6 (demo2 a demo7: demo7-agent-sql-charts.py es
el más completo): un StateGraph con un nodo "agent" que llama al LLM con
las tools bindeadas, un nodo "tools" con ToolNode, y una arista
condicional (tools_condition) que decide si hay que ejecutar una tool o
si ya se puede terminar.

En cada turno el LLM decide si responde directo, si necesita llamar a una
tool, o si tiene que encadenar varias (ver el caso "clima en la capital de
Francia": primero consultar_pais, después consultar_clima con el dato que
devolvió la primera) — eso es lo que arma el loop agent -> tools -> agent.

Requiere en el .env: OPENAI_API_KEY, PINECONE_API_KEY, HF_TOKEN, y haber
corrido antes `python ingest.py` para poblar los tres índices de RAG.
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from config import MODELO
from tools_domain import buscar_libros, buscar_peliculas, buscar_recetas
from tools_external import consultar_clima, consultar_pais

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

TOOLS = [
    buscar_peliculas,
    buscar_libros,
    buscar_recetas,
    consultar_clima,
    consultar_pais,
]

SYSTEM_PROMPT = """Sos el asistente virtual de una plataforma de entretenimiento
y consulta de información general. Podés responder sobre 5 dominios, cada
uno con su propia herramienta:

- Películas, Libros y Recetas: se responden con las tools de búsqueda
  (buscar_peliculas, buscar_libros, buscar_recetas), que consultan una
  base de conocimiento vectorial. Basá tu respuesta ÚNICAMENTE en lo que
  devuelvan esas tools.
- Clima: usá la tool consultar_clima con el nombre de una ciudad.
- Países: usá la tool consultar_pais con el nombre de un país.

Si la pregunta combina dos dominios (por ejemplo "¿cómo está el clima en
la capital de Francia?"), encadená las tools: primero consultar_pais para
obtener la capital, y con ese resultado llamá a consultar_clima.

Reglas importantes:
1. No inventes información. Si ninguna tool trae datos suficientes para
   responder con confianza, decilo honestamente, por ejemplo:
   "No cuento con información suficiente para responder esa consulta."
   o "La información solicitada no se encuentra disponible en mi base de
   conocimiento ni puede obtenerse mediante las herramientas disponibles."
2. No uses tu conocimiento general para responder preguntas de estos 5
   dominios sin pasar por la tool correspondiente (aunque creas saber la
   respuesta) — la fuente de verdad son las tools.
3. Para preguntas totalmente fuera de estos 5 dominios (por ejemplo
   historia, deportes, actualidad), reconocé que no disponés de esa
   información en vez de responder con conocimiento general.
4. Respondé siempre en español, en lenguaje natural y de forma breve.
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(api_key=openai_key, model=MODELO, max_tokens=1024)
llm_with_tools = llm.bind_tools(TOOLS)


def agent_node(state: AgentState):
    """Llama al LLM con el historial actual de mensajes."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools=TOOLS)

graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("agent", END)

graph = graph_builder.compile()


def _estado_inicial(mensaje: str):
    return {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=mensaje),
        ]
    }


def preguntar(mensaje: str) -> str:
    """Ejecuta una consulta suelta contra el agente y devuelve la respuesta final."""
    resultado = graph.invoke(_estado_inicial(mensaje))
    return resultado["messages"][-1].content


def preguntar_con_detalle(mensaje: str) -> str:
    """
    Igual que preguntar(), pero además va imprimiendo, nodo por nodo, qué
    tool eligió el agente y con qué argumentos (mismo estilo de debug que
    usa el profe en demo7-agent-sql-charts.py). Útil para mostrar en la
    entrega que la selección de herramientas es automática.
    """
    ultimo_contenido = ""
    for step in graph.stream(_estado_inicial(mensaje)):
        node_name = list(step.keys())[0]
        last_msg = step[node_name]["messages"][-1]

        print(f"\n--- Nodo: {node_name} ({type(last_msg).__name__}) ---")
        if getattr(last_msg, "tool_calls", None):
            for call in last_msg.tool_calls:
                print(f"Tool: {call['name']}")
                print(f"Args: {call['args']}")
        else:
            print(last_msg.content)
            ultimo_contenido = last_msg.content

    return ultimo_contenido
