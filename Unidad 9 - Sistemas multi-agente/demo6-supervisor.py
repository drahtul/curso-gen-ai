from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.agents import create_agent
from langgraph.types import Command
from _graph_utils import print_graph
from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=openai_key, model="gpt-4.1-nano")


KNOWLEDGE_BASE = {
    "langgraph": "LangGraph models agents as state machines: nodes, edges and a shared state.",
    "supervisor": "Supervisor: a coordinator routes work to specialists and keeps control of the flow.",
    "swarm": "Swarm: agents hand control to each other directly, with no coordinator.",
    "mcp": "MCP connects agents to tools, allowing them to act autonomously and collaboratively.",
}


@tool
def search_kb(topic: str) -> str:
    """Search the course knowledge base for a topic (langgraph, supervisor, swarm, mcp)."""
    key = topic.lower().strip()
    hits = [text for name, text in KNOWLEDGE_BASE.items() if name in key or key in name]
    return "\n".join(hits) if hits else f"No entry for '{topic}'."


researcher_agent = create_agent(
    model=llm,
    tools=[search_kb],
    system_prompt=(
        "You are a researcher. Use search_kb to collect facts about the request. "
        "Report the facts as bullet points. Do NOT write the final article."
    ),
)

writer_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=(
        "You are a technical writer. Using the facts already in the conversation, "
        "write a clear paragraph (max 80 words) answering the user's request."
    ),
)


class Route(BaseModel):
    next: Literal["researcher", "writer", "FINISH"] = Field(
        description="Who should act next, or FINISH when the answer is complete."
    )
    reason: str = Field(description="One short sentence explaining the choice.")


SUPERVISOR_PROMPT = (
    "You are a supervisor coordinating two workers: 'researcher' (gathers facts) "
    "and 'writer' (writes the final answer). Given the conversation so far, decide "
    "who acts next. Route to researcher first if there are no facts yet, then to "
    "writer. Answer FINISH once the writer has produced the final text."
)

supervisor_llm = llm.with_structured_output(Route)


def supervisor(state: MessagesState) -> Command:
    decision = supervisor_llm.invoke([("system", SUPERVISOR_PROMPT)] + state["messages"])
    print(f"  [supervisor] next={decision.next} ({decision.reason})")

    if decision.next == "FINISH":
        return Command(goto=END)
    return Command(goto=decision.next)


def researcher(state: MessagesState) -> Command:
    result = researcher_agent.invoke({"messages": state["messages"]})
    content = result["messages"][-1].content
    print(f"  [researcher] {content[:120]}...")
    return Command(
        update={"messages": [AIMessage(content=content, name="researcher")]},
        goto="supervisor",
    )


def writer(state: MessagesState) -> Command:
    result = writer_agent.invoke({"messages": state["messages"]})
    content = result["messages"][-1].content
    print(f"  [writer] {content[:120]}...")
    return Command(
        update={"messages": [AIMessage(content=content, name="writer")]},
        goto="supervisor",
    )


builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor, destinations=("researcher", "writer", END))
builder.add_node("researcher", researcher, destinations=("supervisor",))
builder.add_node("writer", writer, destinations=("supervisor",))
builder.add_edge(START, "supervisor")

graph = builder.compile()


def run(user_input: str):
    print("\n--- trace ---")
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        {"recursion_limit": 15},
    )
    print(f"\nFinal answer:\n{result['messages'][-1].content}\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Demo 5 - Supervisor clásico")
    print("=" * 80)

    print_graph(graph)
    print("\nEvery arrow goes back to the supervisor: it never loses control.")
    print("Try: 'Explain the difference between MCP and A2A'")
    print("Type 'exit' to quit\n")

    while True:
        user_query = input("Enter your query: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        if not user_query:
            print("Please enter a valid query.\n")
            continue

        run(user_query)
