import operator
from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from _graph_utils import print_graph
from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=openai_key, model="gpt-4.1-nano")


SPECIALISTS = {
    "billing": "You are a billing specialist. Answer only the billing part of the question, in 2 sentences.",
    "technical": "You are a technical support engineer. Answer only the technical part, in 2 sentences.",
    "legal": "You are a legal advisor. Answer only the legal/privacy part, in 2 sentences.",
}


class Classification(BaseModel):
    specialists: List[Literal["billing", "technical", "legal"]] = Field(
        description="All the specialists needed. Use more than one when the question mixes domains."
    )


class RouterState(TypedDict):
    # answers usa un reducer aditivo porque varios especialistas escriben en
    # paralelo; sin el reducer una respuesta podria reemplazar a otra.
    question: str
    specialists: list
    answers: Annotated[list, operator.add]
    final: str


class WorkerInput(TypedDict):
    question: str
    specialist: str

# El router clasifica la pregunta y decide que especialistas deben responder. 
# Luego, se ejecutan en paralelo los estados de cada especialista y finalmente se sintetizan las respuestas en una sola.
def classify(state: RouterState) -> dict:
    # La salida estructurada limita los dominios validos antes de crear las ejecuciones paralelas.
    decision = llm.with_structured_output(Classification).invoke(
        [
            ("system", "Classify which specialists (billing, technical, legal) must answer."),
            ("user", state["question"]),
        ]
    )
    print(f"  [router] -> {decision.specialists}")
    return {"specialists": decision.specialists}

# fan_out ejecuta un estado por cada especialista, creando un flujo paralelo de ejecuciones.
def fan_out(state: RouterState):
    # Send crea estados parciales dinamicos para el mismo nodo specialist;
    # se diferencia de Command, que dirige una ejecucion a un destino
    return [
        Send("specialist", {"question": state["question"], "specialist": name})
        for name in state["specialists"]
    ]


def specialist(state: WorkerInput) -> dict:
    # Cada especialista trabaja de forma independiente sobre la misma pregunta
    # y aporta una respuesta que luego consumira el sintetizador.
    name = state["specialist"]
    response = llm.invoke([("system", SPECIALISTS[name]), ("user", state["question"])])
    print(f"  [{name}] {response.content[:90]}...")
    return {"answers": [f"{name}: {response.content}"]}


def synthesize(state: RouterState) -> dict:
    # Esta etapa reduce las respuestas del fan-out a una salida coherente; es
    # otra llamada generativa y puede introducir errores al fusionar criterios.
    joined = "\n".join(state["answers"])
    response = llm.invoke(
        [
            ("system", "Merge these specialist answers into one coherent reply (max 100 words)."),
            ("user", f"Question: {state['question']}\n\nAnswers:\n{joined}"),
        ]
    )
    return {"final": response.content}


builder = StateGraph(RouterState)
builder.add_node("classify", classify)
builder.add_node("specialist", specialist)
builder.add_node("synthesize", synthesize)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", fan_out, ["specialist"])
builder.add_edge("specialist", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()


def run(user_input: str):
    print("\n--- trace ---")
    result = graph.invoke({"question": user_input, "specialists": [], "answers": [], "final": ""})
    print(f"\nFinal answer:\n{result['final']}\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Demo 7 - Router con ejecuciones paralelas")
    print("=" * 80)

    print_graph(graph)
    print("\nEjemplo: 'You charged me twice, the app also crashes, and can I get my data deleted?'")
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
