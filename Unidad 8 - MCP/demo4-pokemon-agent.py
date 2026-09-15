import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools # type: ignore
from langchain_openai import ChatOpenAI
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
mcp_api_key = os.getenv("POKEMON_MCP_API_KEY")

MCP_SERVER_URL = "https://parliamentary-gray-snipe.fastmcp.app/mcp"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


class PokemonMCPAgent:
    def __init__(self):
        self.llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4.1-nano")
        self.llm_with_tools = None
        self.graph = None
        self.session = None
        self.tools = []
        self._exit_stack = AsyncExitStack()

    async def initialize(self):
        """Initialize MCP session and load available tools."""
        read, write, _ = await self._exit_stack.enter_async_context(
            streamablehttp_client(
                MCP_SERVER_URL,
                headers={"Authorization": f"Bearer {mcp_api_key}"},
            )
        )
        self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

        self.tools = await load_mcp_tools(self.session)

        print(f"Found {len(self.tools)} tools from Pokemon MCP:")
        for tool in self.tools:
            print(f"  - {tool.name}")

    async def close(self):
        """Close the MCP session."""
        await self._exit_stack.aclose()

    def build_graph(self):
        """Build the LangGraph agent."""
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        def agent_node(state: AgentState):
            """Call the LLM with the current messages."""
            messages = state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        tool_node = ToolNode(tools=self.tools)

        graph_builder = StateGraph(AgentState)

        graph_builder.add_node("agent", agent_node)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges("agent", tools_condition)
        graph_builder.add_edge("tools", "agent")
        graph_builder.add_edge("agent", END)

        self.graph = graph_builder.compile()
        print("\nAgent graph built successfully!")

    async def stream_tool_responses(self, user_input: str):
        """Stream responses from the agent."""
        async for step in self.graph.astream({"messages": [HumanMessage(content=user_input)]}):
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


async def main():
    """Main entry point."""
    agent = PokemonMCPAgent()

    try:
        await agent.initialize()

        agent.build_graph()

        print("\n" + "=" * 80)
        print("Pokemon MCP Agent with LangGraph")
        print("=" * 80)

        while True:
            user_query = input("Enter your query: ").strip()
            if user_query.lower() == "exit":
                print("Goodbye!")
                break
            if not user_query:
                print("Please enter a valid query.\n")
                continue

            await agent.stream_tool_responses(user_query)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
