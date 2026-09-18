import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from _graph_utils import print_graph


class AgentState(TypedDict):
    ticket: str
    department: str
    log: Annotated[list, operator.add]


def triage(state: AgentState) -> Command:
    text = state["ticket"].lower()
    if any(word in text for word in ["invoice", "payment", "refund"]):
        target = "billing"
    elif any(word in text for word in ["error", "crash", "bug", "500"]):
        target = "engineering"
    else:
        target = "general"

    return Command(
        update={"department": target, "log": [f"triage -> {target}"]},
        goto=target,
    )


def billing(state: AgentState) -> Command:
    return Command(update={"log": ["billing handled the ticket"]}, goto=END)


def engineering(state: AgentState) -> Command:
    return Command(update={"log": ["engineering handled the ticket"]}, goto=END)


def general(state: AgentState) -> Command:
    return Command(update={"log": ["general support handled the ticket"]}, goto=END)


builder = StateGraph(AgentState)
# Declaring the possible targets is what lets LangGraph draw the edges.
builder.add_node("triage", triage, destinations=("billing", "engineering", "general"))
builder.add_node("billing", billing)
builder.add_node("engineering", engineering)
builder.add_node("general", general)
builder.add_edge(START, "triage")

graph = builder.compile()


if __name__ == "__main__":
    print("=" * 80)
    print("Demo 2 - Command (update + goto)")
    print("=" * 80)

    print_graph(graph)
    print("\nNo add_conditional_edges anywhere: the node itself chose the target.\n")

    tickets = [
        "My invoice is wrong, I need a refund",
        "The app crashes with error 500 on login",
        "What are your opening hours?",
    ]

    for ticket in tickets:
        result = graph.invoke({"ticket": ticket, "department": "", "log": []})
        print("-" * 80)
        print(f"Ticket     : {ticket}")
        print(f"Department : {result['department']}")
        print(f"Log        : {result['log']}")
