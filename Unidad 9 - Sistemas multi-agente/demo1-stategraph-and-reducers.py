import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from _graph_utils import print_graph


class AgentState(TypedDict):
    topic: str
    steps: Annotated[list, operator.add]
    cost: Annotated[int, operator.add]


def classify(state: AgentState) -> dict:
    """First node: decides the topic and records a step."""
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
    return "technical_node" if state["topic"] == "technical" else "general_node"


builder = StateGraph(AgentState)
builder.add_node("classify", classify)
builder.add_node("technical_node", technical_node)
builder.add_node("general_node", general_node)
builder.add_node("summarize", summarize)

builder.add_edge(START, "classify")
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

        for step in graph.stream({"topic": user_input, "steps": [], "cost": 0}):
            print(f"  update -> {step}")

        final = graph.invoke({"topic": user_input, "steps": [], "cost": 0})
        print(f"  final state -> {final}")

    print("\nNotice: 'topic' was overwritten, 'steps' was appended, 'cost' was summed.")
