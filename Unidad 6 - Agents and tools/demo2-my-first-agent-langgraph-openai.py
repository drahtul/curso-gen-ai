from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4.1-nano")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


tools = [multiply, add]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """Call the LLM with the current messages."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("agent", END)

graph = graph_builder.compile()

print("=" * 80)
print("GRAPH STRUCTURE")
print("=" * 80)
print(graph.get_graph().draw_ascii())
print()

def stream_tool_responses(user_input: str):
    for step in graph.stream({"messages": [HumanMessage(content=user_input)]}):
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
user_query = "Suma 35 mas 20. Despues de haber sumado, solo al resultado de esa suma multiplicalo por 2"
stream_tool_responses(user_query)