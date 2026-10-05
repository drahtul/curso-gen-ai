import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from _graph_utils import print_graph


class AgentState(TypedDict):
    # El estado compartido es el contrato entre nodos. Cada reducer define
    # como combinar la actualizacion de un nodo con el valor existente.
    topic: str
    # topic se sobrescribe; steps se concatena y cost se suma mediante add.
    steps: Annotated[list, operator.add]
    cost: Annotated[int, operator.add]


def classify(state: AgentState) -> dict:
    """First node: decides the topic and records a step."""
    # Este routing es determinista: representa una etapa especializada del
    # workflow, no una decision autonoma generada por un LLM.
    topic = "technical" if "error" in state["topic"].lower() else "general"
    return {"topic": topic, "steps": ["classify"], "cost": 1}


def technical_node(state: AgentState) -> dict:
    return {"steps": ["technical_node"], "cost": 5}


def general_node(state: AgentState) -> dict:
    return {"steps": ["general_node"], "cost": 2}


def summarize(state: AgentState) -> dict:
    return {"steps": ["summarize"], "cost": 1}


def route(state: AgentState) -> str:
    """A conditional edge is just a function returning the name of the next node."""
    # La arista condicional separa la decision de routing de la ejecucion del
    # nodo especialista (technical_node o general_node). Esto permite que la decision de routing
    # pueda ser tomada por un LLM, una herramienta externa o un algoritmo local determinista
    return "technical_node" if state["topic"] == "technical" else "general_node"

builder = StateGraph(AgentState)
builder.add_node("classify", classify)
builder.add_node("technical_node", technical_node)
builder.add_node("general_node", general_node)
builder.add_node("summarize", summarize)

builder.add_edge(START, "classify")
# La arista condicional permite que el flujo se bifurque en dos nodos distintos
# dependiendo del valor de topic. La funcion route es determinista, pero podria
# ser reemplazada por una funcion que tome decisiones basadas en la salida de un LLM.
builder.add_conditional_edges("classify", route, ["technical_node", "general_node"])
builder.add_edge("technical_node", "summarize")
builder.add_edge("general_node", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()


if __name__ == "__main__":
    print("=" * 80)
    print("Demo 1 - StateGraph, reducers y aristas condicionales")
    print("=" * 80)

    print_graph(graph)

    for user_input in ["I get an error 500 when deploying", "How much does the course cost?"]:
        print("\n" + "-" * 80)
        print(f"Input: {user_input}")

        # stream permite observar actualizaciones intermedias; invoke, usado
        # despues, devuelve el estado final del mismo tipo de ejecucion.
        for step in graph.stream({"topic": user_input, "steps": [], "cost": 0}):
            print(f"  update -> {step}")

        final = graph.invoke({"topic": user_input, "steps": [], "cost": 0})
        print(f"  final state -> {final}")

    print("\nNotice: 'topic' was overwritten, 'steps' was appended, 'cost' was summed.")
