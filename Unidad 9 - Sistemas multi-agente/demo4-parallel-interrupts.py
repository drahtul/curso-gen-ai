from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

checkpointer = InMemorySaver()   # short-term: one conversation per thread_id


def merge_dicts(current: dict, update: dict) -> dict:
    """Reducer: merge the approvals collected by each parallel branch."""
    return {**current, **update}


class AgentState(TypedDict):
    request: str
    approvals: Annotated[dict, merge_dicts]
    outcome: str


def manager_review(state: AgentState) -> dict:
    """Runs in parallel with finance_review; pauses waiting for the manager's call."""
    decision = interrupt(
        {"question": f"[Manager] Approve expense '{state['request']}'? (approve/reject)"}
    )
    return {"approvals": {"manager": decision}}


def finance_review(state: AgentState) -> dict:
    """Runs in parallel with manager_review; pauses waiting for finance's call."""
    decision = interrupt(
        {"question": f"[Finance] Approve expense '{state['request']}'? (approve/reject)"}
    )
    return {"approvals": {"finance": decision}}


def finalize(state: AgentState) -> dict:
    """Runs only after BOTH parallel interrupts have been resolved."""
    approvals = state["approvals"]
    if all(decision == "approve" for decision in approvals.values()):
        outcome = f"APPROVED by everyone: {approvals}"
    else:
        outcome = f"REJECTED, not everyone approved: {approvals}"
    return {"outcome": outcome}


builder = StateGraph(AgentState)
builder.add_node("manager_review", manager_review)
builder.add_node("finance_review", finance_review)
builder.add_node("finalize", finalize)

# Fan-out: both reviews start at the same time...
builder.add_edge(START, "manager_review")
builder.add_edge(START, "finance_review")
# ...and fan-in: "finalize" waits until BOTH branches have produced a result.
builder.add_edge("manager_review", "finalize")
builder.add_edge("finance_review", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile(checkpointer=checkpointer)


def ask(request: str, thread_id: str):
    """Run one turn, resolving as many *simultaneous* interrupts as the graph raises."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"request": request, "approvals": {}}, config)

    while "__interrupt__" in result:
        pending = result["__interrupt__"]
        print(f"\n--- HUMAN IN THE LOOP: {len(pending)} pending approval(s) ---")

        # Each pending interrupt has its own id: collect one decision per id.
        resume_map = {}
        for pending_interrupt in pending:
            question = pending_interrupt.value["question"]
            decision = input(f"{question}: ").strip()
            resume_map[pending_interrupt.id] = decision

        # A single resume can answer several interrupts at once via {id: value}.
        result = graph.invoke(Command(resume=resume_map), config)

    print(f"\nOutcome: {result['outcome']}\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Demo 4 - Human-in-the-loop con varios interrupts")
    print("=" * 80)
    print("An expense needs approval from both Manager and Finance, at the same time.")
    print("Both interrupts are raised together; you resolve both before the graph continues.")
    print("Type 'exit' to quit.\n")

    thread = "conversation-1"
    while True:
        user_request = input("Describe the expense to submit: ").strip()
        if user_request.lower() == "exit":
            print("Goodbye!")
            break
        if not user_request:
            print("Please enter a valid request.\n")
            continue

        ask(user_request, thread)
