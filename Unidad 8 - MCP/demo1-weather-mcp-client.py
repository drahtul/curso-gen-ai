import asyncio
import json
from fastmcp import Client

async def main():
    client = Client("demo1-weather-mcp-server.py")
    async with client:
        
        result = await client.list_tools()
        result_dicts = [tool.__dict__ for tool in result]
        print(json.dumps(result_dicts, indent=2))

        forecast = await client.call_tool("get_forecast", {"latitude": 40.7128, "longitude": -74.0060})
        print("\nForecast Result:")
        # print(forecast.data.result)
        print(forecast.content[0].text)
        
asyncio.run(main())