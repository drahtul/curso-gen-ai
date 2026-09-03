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
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


tools = [multiply, add]
agent = create_agent(model=llm, tools=tools)


def stream_tool_responses(user_input: str):
    # debug con streaming de respuestas del agente
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
# condicionar secuencia de operaciones, primero sumar y luego multiplicar el resultado
# pero de la manera que está instanciado el agente no puedo condicionarlo 
user_query = "Suma 35 mas 20. Despues de haber sumado, solo al resultado de esa suma multiplicalo por 2"
stream_tool_responses(user_query)