from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import InjectedState
from langgraph.checkpoint.memory import InMemorySaver
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
    "mcp": "MCP connects agents to tools.",
}


@tool
def search_kb(topic: str) -> str:
    """Search the course knowledge base for a topic (langgraph, supervisor, swarm, a2a, mcp)."""
    key = topic.lower().strip()
    hits = [text for name, text in KNOWLEDGE_BASE.items() if name in key or key in name]
    return "\n".join(hits) if hits else f"No entry for '{topic}'."


class SwarmState(MessagesState):
    active_agent: str


def create_handoff_tool(agent_name: str, description: str):
    """Build a tool whose only effect is to transfer control to another agent."""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name, description=description)
    def handoff(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Control transferred to {agent_name}.",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        print(f"  [handoff] -> {agent_name}")
        
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={"messages": state["messages"] + [tool_message], "active_agent": agent_name},
        )

    return handoff


to_writer = create_handoff_tool("writer", "Hand the collected facts over to the writer agent.")
to_researcher = create_handoff_tool("researcher", "Hand control to the researcher agent to gather facts.")

researcher_agent = create_agent(
    model=llm,
    tools=[search_kb, to_writer],
    system_prompt=(
        "You are the researcher of a two-agent team. Use search_kb to gather facts. "
        "As soon as you have the facts, call transfer_to_writer. Never write the final text yourself."
    ),
    name="researcher",
)

writer_agent = create_agent(
    model=llm,
    tools=[to_researcher],
    system_prompt=(
        "You are the writer of a two-agent team. Using the facts in the conversation, write a "
        "clear paragraph (max 80 words). If facts are missing, call transfer_to_researcher instead."
    ),
    name="writer",
)


def entry(state: SwarmState) -> Command:
    target = state.get("active_agent") or "researcher"
    print(f"  [entry] active agent = {target}")
    return Command(goto=target, update={"active_agent": target})


builder = StateGraph(SwarmState)
builder.add_node("entry", entry, destinations=("researcher", "writer"))
builder.add_node("researcher", researcher_agent, destinations=("writer", END))
builder.add_node("writer", writer_agent, destinations=("researcher", END))
builder.add_edge(START, "entry")
builder.add_edge("researcher", END)
builder.add_edge("writer", END)

graph = builder.compile(checkpointer=InMemorySaver())


def run(user_input: str, thread_id: str = "swarm"):
    print("\n--- trace ---")
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        {"configurable": {"thread_id": thread_id}, "recursion_limit": 15},
    )
    print(f"\nActive agent after the turn: {result['active_agent']}")
    print(f"\nFinal answer:\n{result['messages'][-1].content}\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Demo 8 - Swarm (red descentralizada)")
    print("=" * 80)

    print_graph(graph)
    print("\nAgents point at each other. Ask a follow-up to see the writer stay in charge.")
    print("Try: 'Explain the difference between supervisor and swarm' then 'make it shorter'")
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
