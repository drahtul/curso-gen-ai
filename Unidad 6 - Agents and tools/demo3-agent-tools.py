from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from langgraph.graph.message import add_messages
import requests
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

@tool
def count_r_in_word(word: str) -> int:
    """Count how many 'r' letters are in the given word."""
    return word.lower().count('r')

@tool
def weather_tool(city: str) -> str:
    """
    Retrieve current weather for a city using Open-Meteo.
    """
    # First, get coordinates from Open-Meteo geocoding API
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    geo_resp = requests.get(geo_url).json()

    if "results" not in geo_resp or len(geo_resp["results"]) == 0:
        return f"Could not find coordinates for {city}"

    lat = geo_resp["results"][0]["latitude"]
    lon = geo_resp["results"][0]["longitude"]

    # Get current weather
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    weather_resp = requests.get(weather_url).json()

    if "current_weather" not in weather_resp:
        return f"Weather data unavailable for {city}"

    weather = weather_resp["current_weather"]
    temp = weather["temperature"]
    wind = weather["windspeed"]
    condition = weather.get("weathercode", "unknown")

    return f"Current weather in {city}: {temp}°C, wind {wind} km/h, condition code {condition}"


@tool
def convert_temperature(celsius: float, to_fahrenheit: bool = True) -> float:
    """
    Convert temperature between Celsius and Fahrenheit.
    If to_fahrenheit=True, converts Celsius to Fahrenheit.
    If to_fahrenheit=False, converts Fahrenheit to Celsius.
    Returns the converted temperature.
    """
    if to_fahrenheit:
        fahrenheit = (celsius * 9/5) + 32
        return round(fahrenheit, 2)
    else:
        celsius_result = (celsius - 32) * 5/9
        return round(celsius_result, 2)


@tool
def analyze_text(text: str) -> dict:
    """
    Analyze text and return statistics: word count, character count,
    character count without spaces, and average word length.
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    char_count_no_spaces = len(text.replace(" ", ""))
    avg_word_length = char_count_no_spaces / word_count if word_count > 0 else 0

    return {
        "word_count": word_count,
        "character_count": char_count,
        "character_count_no_spaces": char_count_no_spaces,
        "average_word_length": round(avg_word_length, 2)
    }


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4.1-nano")

tools = [count_r_in_word, weather_tool, convert_temperature, analyze_text]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """Call the LLM with the current messages."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("agent", END)

# Compile the graph
graph = graph_builder.compile()

def stream_tool_responses(user_input: str):
    for step in graph.stream({"messages": [HumanMessage(content=user_input)]}):
        print("\n--- Node Output ---")
        node_name = list(step.keys())[0]
        print(f"Node: {node_name}")
        state = step[node_name]

        # Show the last message in detail
        last_msg = state["messages"][-1]
        print(f"Message type: {type(last_msg).__name__}")

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            print(f"Tool calls: {last_msg.tool_calls}")
        else:
            print(f"Content: {last_msg.content}")
    print()

print("=" * 80)
print("Test 1: Weather and count letters")
print("=" * 80)
user_query = "How is the weather in Madrid and how many r are in 'love'?"
stream_tool_responses(user_query)

print("=" * 80)
print("Test 2: Temperature conversion and text analysis")
print("=" * 80)
user_query = "Convert 72 Fahrenheit to Celsius and analyze this text: 'The quick brown fox jumps over the lazy dog'"
stream_tool_responses(user_query)