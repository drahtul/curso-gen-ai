from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=openai_key, model="gpt-4.1-nano")

checkpointer = InMemorySaver()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str
    approved: bool


def write_query(state: AgentState) -> dict:
    system = (
        "You are a database assistant. Translate the last user request into a single "
        "SQL statement against a 'customers(id, name, email, plan)' table. "
        "Reply with ONLY the SQL statement, no explanation."
    )
    response = llm.invoke([("system", system)] + state["messages"])
    return {"draft": response.content}


def human_review(state: AgentState) -> Command:
    """Pause the graph: a human must approve, reject or edit the query before it runs."""
    decision = interrupt(
        {
            "question": "Run this query? (approve / reject / or type a replacement query)",
            "draft": state["draft"],
        }
    )

    if decision == "approve":
        return Command(update={"approved": True}, goto="execute")
    if decision == "reject":
        return Command(
            update={"approved": False, "messages": [AIMessage(content="(query rejected, nothing was run)")]},
            goto=END,
        )
    # Anything else is treated as a human-edited query.
    return Command(update={"draft": decision, "approved": True}, goto="execute")


def execute(state: AgentState) -> dict:
    """'Execute' the approved query (simulated) and report back."""
    return {"messages": [AIMessage(content=f"Executed: {state['draft']}")]}


builder = StateGraph(AgentState)
builder.add_node("write_query", write_query)
builder.add_node("human_review", human_review, destinations=("execute", END))
builder.add_node("execute", execute)
builder.add_edge(START, "write_query")
builder.add_edge("write_query", "human_review")
builder.add_edge("execute", END)

graph = builder.compile(checkpointer=checkpointer)


def ask(user_input: str, thread_id: str):
    """Run one turn, handling the interrupt interactively."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n--- HUMAN IN THE LOOP ---")
        print(f"Draft query: {payload['draft']}")
        decision = input(f"{payload['question']}: ").strip()
        result = graph.invoke(Command(resume=decision), config)

    print(f"\nAgent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Demo 3 - Human-in-the-loop")
    print("=" * 80)
    print("Describe what you want in plain English; every query needs approval before running.")
    print("Type 'exit' to quit.\n")

    thread = "conversation-1"
    while True:
        user_query = input(f"[thread={thread}] Enter your request: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        if not user_query:
            print("Please enter a valid request.\n")
            continue

        ask(user_query, thread)
