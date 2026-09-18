import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from _graph_utils import print_graph
from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=openai_key, model="gpt-4o-mini")

MAX_ITERATIONS = 3

CRITERIA = (
    "1) at most 60 words, "
    "2) no corporate clichés (e.g. 'we apologize for any inconvenience', 'your feedback is valuable'), "
    "3) includes one concrete next step for the customer"
)


class Evaluation(BaseModel):
    """The evaluator's verdict."""

    passed: bool = Field(description="True only if every criterion is satisfied.")
    feedback: str = Field(description="Concrete, actionable feedback for the writer.")


class AgentState(TypedDict):
    complaint: str
    draft: str
    feedback: str
    passed: bool
    iterations: Annotated[int, operator.add]
    history: Annotated[list, operator.add]


def generate(state: AgentState) -> dict:
    if state.get("feedback"):
        user = (
            f"Customer complaint: {state['complaint']}\n"
            f"Previous reply: {state['draft']}\n"
            f"Fix this feedback: {state['feedback']}"
        )
    else:
        user = f"Customer complaint: {state['complaint']}"

    response = llm.invoke(
        [("system", "You write customer support replies."), ("user", user)]
    )
    print(f"  [generate] {response.content}")
    return {"draft": response.content, "iterations": 1, "history": [response.content]}


def evaluate(state: AgentState) -> dict:
    verdict = llm.with_structured_output(Evaluation).invoke(
        [
            ("system", f"You are a strict editor. Criteria: {CRITERIA}"),
            ("user", state["draft"]),
        ]
    )
    print(f"  [evaluate] passed={verdict.passed} :: {verdict.feedback}")
    return {"passed": verdict.passed, "feedback": verdict.feedback}


def should_continue(state: AgentState) -> Literal["generate", "__end__"]:
    """Loop back only if we failed AND we still have budget."""
    if state["passed"] or state["iterations"] >= MAX_ITERATIONS:
        return "__end__"
    return "generate"


builder = StateGraph(AgentState)
builder.add_node("generate", generate)
builder.add_node("evaluate", evaluate)
builder.add_edge(START, "generate")
builder.add_edge("generate", "evaluate")
builder.add_conditional_edges("evaluate", should_continue, ["generate", END])

graph = builder.compile()


def run(complaint: str):
    print("\n--- trace ---")
    result = graph.invoke(
        {"complaint": complaint, "draft": "", "feedback": "", "passed": False, "iterations": 0, "history": []}
    )
    print(f"\nIterations: {result['iterations']} | passed: {result['passed']}")
    print(f"\nFinal text:\n{result['draft']}\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Demo 5 - Evaluator-optimizer")
    print("=" * 80)

    print_graph(graph)
    print(f"\nCriteria: {CRITERIA}")
    print("Try: 'My order arrived 2 weeks late and the box was crushed'")
    print("Type 'exit' to quit\n")

    while True:
        user_query = input("Enter a customer complaint: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        if not user_query:
            print("Please enter a valid complaint.\n")
            continue

        run(user_query)
