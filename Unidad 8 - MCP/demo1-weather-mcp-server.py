from typing import Any

import httpx
from fastmcp import FastMCP
# from mcp.server.fastmcp import FastMCP

# FastMCP publica un contrato de herramientas que un cliente puede descubrir;
# MCP estandariza la interfaz, mientras NWS sigue siendo la API externa.
mcp = FastMCP("weather")

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    # Esta funcion es un helper HTTP interno del servidor. El cliente MCP no
    # necesita conocer URLs ni detalles de autenticacion de NWS.
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
            Event: {props.get("event", "Unknown")}
            Area: {props.get("areaDesc", "Unknown")}
            Severity: {props.get("severity", "Unknown")}
            Description: {props.get("description", "No description available")}
            Instructions: {props.get("instruction", "No specific instructions provided")}
            """

@mcp.tool()
async def get_alerts(state: str) -> str:
    # El decorador convierte la funcion Python en una herramienta MCP con
    # nombre, descripcion y esquema de entrada para clientes y modelos.
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    # Otra herramienta MCP encapsula dos llamadas REST encadenadas y entrega
    # al agente un resultado ya preparado para incluirse en el contexto.
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # Primero se obtiene el endpoint de la grilla meteorologica.
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # La respuesta anterior determina la URL de la prevision detallada.
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Se transforma la respuesta externa en texto legible para reducir ruido
    # antes de que el LLM la interprete.
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    # Limitar los periodos controla el tamano del contexto y el costo de salida.
    for period in periods[:5]:
        forecast = f"""
                    {period["name"]}:
                    Temperature: {period["temperature"]}°{period["temperatureUnit"]}
                    Wind: {period["windSpeed"]} {period["windDirection"]}
                    Forecast: {period["detailedForecast"]}
                    """
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)

def main():
    mcp.run()

if __name__ == "__main__":
    main()