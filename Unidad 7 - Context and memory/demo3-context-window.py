from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
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
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    geo_resp = requests.get(geo_url).json()

    if "results" not in geo_resp or len(geo_resp["results"]) == 0:
        return f"Could not find coordinates for {city}"

    lat = geo_resp["results"][0]["latitude"]
    lon = geo_resp["results"][0]["longitude"]

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
    """State containing messages and conversation summary."""
    messages: Annotated[list, add_messages]
    summary: str

llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-5.6-luna")
summarizer_llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4.1-nano")

tools = [count_r_in_word, weather_tool, convert_temperature, analyze_text]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """Call the LLM with recent messages and summary context."""
    messages = state["messages"]
    summary = state["summary"]

    context_messages = []
    if summary:
        context_messages.append(
            SystemMessage(content=f"Previous conversation summary:\n{summary}")
        )

    recent_messages = messages[-5:] if len(messages) > 5 else messages
    context_messages.extend(recent_messages)

    response = llm_with_tools.invoke(context_messages)
    return {"messages": [response]}

def summarizer_node(state: AgentState):
    """Update conversation summary when it gets long.
    This replaces ConversationSummaryMemory logic."""
    messages = state["messages"]
    current_summary = state["summary"]

    if len(messages) > 5:
        conversation_text = "\n".join(
            f"{msg.type}: {msg.content if hasattr(msg, 'content') else str(msg)}"
            for msg in messages[:-5]
        )

        summary_prompt = f"""Given this conversation history, provide a concise summary covering:
                            - Key topics discussed
                            - Important decisions made
                            - Relevant context for future messages

                            Current summary: {current_summary if current_summary else 'None yet'}

                            Conversation:
                            {conversation_text}

                            Provide an updated summary:"""

        new_summary = summarizer_llm.invoke([HumanMessage(content=summary_prompt)]).content

        return {"summary": new_summary}
    else:
        return {"summary": current_summary}

tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("summarizer", summarizer_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "summarizer"})
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("summarizer", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

def stream_tool_responses(user_input: str, thread_id: str):
    """Stream responses with memory managed through agent state and checkpointer.

    - Recent messages (last 5) kept for short-term context
    - Summary maintained for long-term context
    - State persisted via checkpointer (no external state handling needed)
    """
    config = {"configurable": {"thread_id": thread_id}}

    for step in graph.stream(
        {"messages": [HumanMessage(content=user_input)], "summary": ""},
        config
    ):
        print("\n--- Node Output ---")
        node_name = list(step.keys())[0]
        print(f"Node: {node_name}")
        node_state = step[node_name]

        print("\nAgent State:")
        if "messages" in node_state:
            print(f"  Total messages: {len(node_state['messages'])}")

            last_msg = node_state["messages"][-1]
            print(f"  Last message type: {type(last_msg).__name__}")
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                print(f"  Tool calls: {last_msg.tool_calls}")
            else:
                print(f"  Last message content: {last_msg.content}")

        summary = node_state.get("summary", "")
        if summary:
            print(f"  Summary: {summary[:1500]}...")
        else:
            print("  Summary: (empty)")
    print()

if __name__ == "__main__":
    print("=" * 80)
    print("Agent with modern memory management (LangGraph State + Checkpointer)")
    print("=" * 80)
    print("\nThis demo uses the modern LangChain approach:")
    print("- Messages stored in state (no external memory classes)")
    print("- Last 5 messages kept for short-term context (replaces BufferWindow)")
    print("- Automatic summarization when conversation grows (replaces SummaryMemory)")
    print("- Summarizer node maintains summary in state")
    print("- State persisted via checkpointer (no manual state handling)")
    print("\nAvailable tools:")
    print("- Count 'r' letters in a word")
    print("- Get weather for a city")
    print("- Convert temperature (Celsius/Fahrenheit)")
    print("- Analyze text statistics")
    print("\nType 'exit' to quit\n")

    thread_id = "conversation-1"

    while True:
        user_query = input("Enter your query: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        if not user_query:
            print("Please enter a valid query.\n")
            continue

        stream_tool_responses(user_query, thread_id=thread_id)
