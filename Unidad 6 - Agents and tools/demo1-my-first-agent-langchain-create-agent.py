from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key=openai_key,
    model="gpt-4.1-nano",
    max_tokens=1024,
)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    # @tool expone esta funcion al modelo con su nombre, tipos y descripcion;
    # el calculo sigue siendo determinista y lo ejecuta Python.
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    # El LLM decide cuando solicitar la herramienta, pero no reemplaza la
    # ejecucion del codigo que produce el resultado numerico.
    return a + b


tools = [multiply, add]
# create_agent aporta el ciclo de tool calling: mensaje, decision del modelo,
# ejecucion de la tool, devolucion del resultado y respuesta final.
agent = create_agent(model=llm, tools=tools)

# agent.plan("Suma 35 mas 20. Despues de haber sumado, solo al resultado de esa suma multiplicalo por 2")


def stream_tool_responses(user_input: str):
    # El streaming muestra eventos intermedios del agente; no implica
    # necesariamente que cada evento sea un token individual.
    for step in agent.stream({"messages": [HumanMessage(content=user_input)]}):
        print("\n--- Node Output ---")
        node_name = list(step.keys())[0]
        print(f"Node: {node_name}")
        state = step[node_name]

        last_msg = state["messages"][-1]
        print(f"Message type: {type(last_msg).__name__}")

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            print(f"Tool calls: {last_msg.tool_calls}")
        else:
            print(f"Content: {last_msg.content}")
    print()


print("=" * 80)
print("Test 1: Cuanto es 5 multiplicado por 3?")
print("=" * 80)
user_query = "Cuanto es 5 multiplicado por 3?"
stream_tool_responses(user_query)

print("=" * 80)
print("Test 2: Suma 35 y 20, y el resultado multiplicalo por 2")
print("=" * 80)
# Un agente autonomo puede elegir una secuencia razonable, pero esta creacion
# de alto nivel no garantiza un workflow determinista suma -> multiplicacion.
# Para imponer orden hace falta logica adicional o un grafo explicito.
user_query = "Suma 35 mas 20. Despues de haber sumado, solo al resultado de esa suma multiplicalo por 2"
stream_tool_responses(user_query)