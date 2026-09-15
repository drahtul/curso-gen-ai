import asyncio
import json
from fastmcp import Client
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
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


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


class PokemonMCPAgent:
    def __init__(self):
        self.mcp_client = None
        self.mcp_tools = {}
        self.llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4.1-nano")
        self.llm_with_tools = None
        self.graph = None

    async def initialize(self):
        """Initialize MCP client and fetch available tools."""
        self.mcp_client = Client(
            "https://parliamentary-gray-snipe.fastmcp.app/mcp",
            auth=mcp_api_key,
        )

        async with self.mcp_client:
            mcp_tools_list = await self.mcp_client.list_tools()

            for mcp_tool in mcp_tools_list:
                self.mcp_tools[mcp_tool.name] = mcp_tool

            print(f"Found {len(self.mcp_tools)} tools from Pokemon MCP:")
            for tool_name in self.mcp_tools.keys():
                print(f"  - {tool_name}")

    def _create_mcp_tool_wrapper(self, tool_name: str):
        """Create a LangChain tool wrapper for an MCP tool."""
        mcp_tool = self.mcp_tools[tool_name]
        tool_description = mcp_tool.description or f"Call the {tool_name} tool"

        input_schema = getattr(mcp_tool, "inputSchema", None)

        def mcp_tool_call(**kwargs):
            async def async_call():
                async with Client(
                    "https://parliamentary-gray-snipe.fastmcp.app/mcp",
                    auth=mcp_api_key,
                ) as client:
                    result = await client.call_tool(tool_name, kwargs)
                    if result.content and len(result.content) > 0:
                        response_text = result.content[0].text

                        if tool_name == "get_pokemon_info":
                            try:
                                data = json.loads(response_text)
                                filtered_data = {
                                    "name": data.get("name"),
                                    "height": data.get("height"),
                                    "weight": data.get("weight"),
                                    "types": [t["type"]["name"] for t in data.get("types", [])],
                                    "abilities": [a["ability"]["name"] for a in data.get("abilities", [])],
                                    "base_experience": data.get("base_experience")
                                }
                                return json.dumps(filtered_data, indent=2)
                            except (json.JSONDecodeError, KeyError, TypeError):
                                return response_text

                        return response_text
                    return str(result)

            return asyncio.run(async_call())

        # Build the tool with proper schema
        tool_kwargs = {
            "name": tool_name,
            "description": tool_description,
        }

        # If the tool has input schema, use it to define args_schema
        if input_schema:
            from pydantic import create_model
            from typing import Optional, Any

            # Extract properties from JSON schema
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            # Create field definitions for Pydantic model
            field_definitions = {}
            for prop_name, prop_schema in properties.items():
                prop_type = Any
                if prop_schema.get("type") == "string":
                    prop_type = str
                elif prop_schema.get("type") == "integer":
                    prop_type = int
                elif prop_schema.get("type") == "number":
                    prop_type = float
                elif prop_schema.get("type") == "boolean":
                    prop_type = bool

                # Make optional if not required
                if prop_name not in required:
                    prop_type = Optional[prop_type]

                field_definitions[prop_name] = (prop_type, ...)

            # Create the Pydantic model
            if field_definitions:
                args_schema = create_model(f"{tool_name}_args", **field_definitions)
                tool_kwargs["args_schema"] = args_schema

        langchain_tool = StructuredTool.from_function(
            mcp_tool_call,
            **tool_kwargs
        )

        return langchain_tool

    def build_graph(self):
        """Build the LangGraph agent."""
        # Create tool wrappers for all MCP tools
        langchain_tools = []
        for tool_name in self.mcp_tools.keys():
            try:
                langchain_tool = self._create_mcp_tool_wrapper(tool_name)
                langchain_tools.append(langchain_tool)
            except Exception as e:
                print(f"Error creating wrapper for {tool_name}: {e}")

        self.llm_with_tools = self.llm.bind_tools(langchain_tools)

        def agent_node(state: AgentState):
            """Call the LLM with the current messages."""
            messages = state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        tool_node = ToolNode(tools=langchain_tools)

        graph_builder = StateGraph(AgentState)

        graph_builder.add_node("agent", agent_node)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges("agent", tools_condition)
        graph_builder.add_edge("tools", "agent")
        graph_builder.add_edge("agent", END)

        self.graph = graph_builder.compile()
        print("\nAgent graph built successfully!")

    def stream_tool_responses(self, user_input: str):
        """Stream responses from the agent."""
        for step in self.graph.stream({"messages": [HumanMessage(content=user_input)]}):
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

        agent.stream_tool_responses(user_query)


if __name__ == "__main__":
    asyncio.run(main())
