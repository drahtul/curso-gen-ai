from typing import Annotated
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from api_tools import consultar_clima, consultar_pais
from vector_tools import buscar_libros, buscar_peliculas, buscar_recetas
from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """Sos un asistente virtual de una plataforma de entretenimiento e información general.
Respondés siempre en español, de forma breve y concreta.

Reglas de trabajo:
1. Nunca respondas sobre películas, libros o recetas de memoria: consultá siempre
   la base vectorial correspondiente y usá únicamente lo que devuelva. Si la base
   devuelve resultados pero no contienen el dato puntual que se pidió (por ejemplo
   el director o la duración de una película), decilo explícitamente en vez de
   completarlo con lo que sabés.
2. Nunca inventes datos de clima o de países: usá las tools.
3. Si la consulta necesita varios pasos, encadená las tools. Por ejemplo, para
   "el clima en la capital de Francia": primero consultar_pais('France') para
   obtener la capital y después consultar_clima con esa ciudad.
4. consultar_pais espera el nombre del país en inglés, traducilo si hace falta
   (Japón -> Japan, Noruega -> Norway, Brasil -> Brazil).
5. Si una tool devuelve SIN_RESULTADOS o ERROR, no completes con conocimiento
   propio: explicá honestamente que no tenés esa información.
6. Si la consulta no pertenece a ninguno de los cinco dominios (deportes,
   política, historia, cálculos, etc.), no llames a ninguna tool y respondé:
   "No cuento con información suficiente para responder esa consulta. Puedo
   ayudarte con películas, libros, recetas, clima y datos de países."
7. Está prohibido inventar. Ante la duda, admitilo."""

TOOLS = [
    buscar_peliculas,
    buscar_libros,
    buscar_recetas,
    consultar_clima,
    consultar_pais,
]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(
    api_key=openai_key,
    model="gpt-4.1-nano"
)
llm_with_tools = llm.bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict:
    """Nodo razonador: decide si responde o si llama a una tool."""
    mensajes = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(mensajes)]}


def build_graph():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(tools=TOOLS))

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")
    graph_builder.add_edge("agent", END)

    return graph_builder.compile()


graph = build_graph()


def preguntar(consulta: str, verbose: bool = False) -> str:
    """Ejecuta el grafo para una consulta y devuelve la respuesta final."""
    resultado = graph.invoke(
        {"messages": [HumanMessage(content=consulta)]}
    )

    if verbose:
        for mensaje in resultado["messages"]:
            if isinstance(mensaje, AIMessage) and mensaje.tool_calls:
                for llamada in mensaje.tool_calls:
                    print(f"   [tool] {llamada['name']}({llamada['args']})")

    return resultado["messages"][-1].content


def main() -> None:
    print("=" * 80)
    print("Asistente virtual - películas, libros, recetas, clima y países")
    print("=" * 80)
    print("Escribí 'salir' para terminar.\n")

    while True:
        try:
            consulta = input("Consulta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Hasta luego!")
            break

        if consulta.lower() in {"salir", "exit", "quit"}:
            print("¡Hasta luego!")
            break
        if not consulta:
            continue

        print(f"\n{preguntar(consulta, verbose=True)}\n")


if __name__ == "__main__":
    main()
