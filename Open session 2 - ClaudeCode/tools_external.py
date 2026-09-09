"""
Tools externas del agente: clima y países, consultando APIs públicas
en vivo (no bases vectoriales).

Clima: misma lógica que weather-api/server.js de la Unidad 3 (geocoding
+ forecast de Open-Meteo), reescrita en Python en vez de correr el
servidor Node aparte.

Países: la demo de países del profe (Open session 1 / paises-api.js)
usa api.restcountries.com/v5, una API paga que requiere una API key
(RESTCOUNTRIES_API_KEY) que el equipo puede no tener. Para no depender
de esa key usamos la versión pública y gratuita de RestCountries
(v3.1), que expone los mismos datos (capital, moneda, idiomas,
población) sin necesitar credenciales.
"""

import requests
from langchain_core.tools import tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
COUNTRIES_URL = "https://restcountries.com/v3.1/name/{name}"

# Mismo diccionario de códigos de clima que usa weather-api/server.js
WEATHER_CODES = {
    0: "Cielo despejado",
    1: "Mayormente despejado",
    2: "Parcialmente nublado",
    3: "Nublado",
    45: "Niebla",
    48: "Niebla con escarcha",
    51: "Llovizna ligera",
    53: "Llovizna moderada",
    55: "Llovizna intensa",
    61: "Lluvia ligera",
    63: "Lluvia moderada",
    65: "Lluvia intensa",
    71: "Nevada ligera",
    73: "Nevada moderada",
    75: "Nevada intensa",
    80: "Chubascos ligeros",
    81: "Chubascos moderados",
    82: "Chubascos violentos",
    95: "Tormenta",
    96: "Tormenta con granizo ligero",
    99: "Tormenta con granizo intenso",
}


# region Clima

def _obtener_coordenadas(ciudad):
    print(f"  -> [API] GET {GEOCODING_URL}?name={ciudad}")
    response = requests.get(
        GEOCODING_URL,
        params={"name": ciudad, "count": 1, "language": "es", "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    resultados = data.get("results")
    if not resultados:
        return None
    return resultados[0]


@tool
def consultar_clima(ciudad: str) -> str:
    """Consulta el clima actual de una ciudad usando una API pública
    (Open-Meteo). Úsala para responder preguntas sobre clima, temperatura
    o si va a llover en una ciudad determinada (ej: "¿cómo está el clima
    en Montevideo?", "¿va a llover hoy en Madrid?"). El parámetro debe
    ser el nombre de una ciudad, no de un país."""
    ubicacion = _obtener_coordenadas(ciudad)
    if not ubicacion:
        return f'No se encontró la ciudad "{ciudad}".'

    print(f"  -> [API] GET {FORECAST_URL}")
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": ubicacion["latitude"],
            "longitude": ubicacion["longitude"],
            "current_weather": True,
            "timezone": "auto",
        },
        timeout=10,
    )
    response.raise_for_status()
    clima = response.json().get("current_weather")

    if not clima:
        return "No se pudo obtener el clima en este momento."

    descripcion = WEATHER_CODES.get(clima["weathercode"], "Desconocido")
    return (
        f"Clima actual en {ubicacion['name']}, {ubicacion.get('country', '')}: "
        f"{clima['temperature']} °C, {descripcion.lower()}, "
        f"viento de {clima['windspeed']} km/h. (hora local: {clima['time']})"
    )


# endregion

# region Países

@tool
def consultar_pais(nombre: str) -> str:
    """Consulta información pública de un país: capital, moneda, idiomas
    oficiales y población. Úsala para responder preguntas sobre países
    (ej: "¿cuál es la capital de Japón?", "¿qué moneda utiliza Brasil?",
    "¿cuántos habitantes tiene Canadá?"). También es el primer paso
    necesario cuando preguntan por el clima de la capital de un país:
    primero conseguí la capital acá y después consultá consultar_clima
    con esa ciudad."""
    url = COUNTRIES_URL.format(name=nombre)
    print(f"  -> [API] GET {url}")
    response = requests.get(
        url,
        params={"fields": "name,capital,region,population,currencies,languages"},
        timeout=10,
    )

    if response.status_code == 404:
        return f'No se encontró el país "{nombre}".'
    response.raise_for_status()

    data = response.json()
    pais = data[0] if isinstance(data, list) else data

    capital = ", ".join(pais.get("capital", [])) or "desconocida"
    idiomas = ", ".join((pais.get("languages") or {}).values()) or "desconocidos"
    monedas = ", ".join(
        f"{info.get('name')} ({codigo})"
        for codigo, info in (pais.get("currencies") or {}).items()
    ) or "desconocida"

    return (
        f"{pais['name']['common']}: capital {capital}, "
        f"región {pais.get('region', 'desconocida')}, "
        f"población {pais.get('population', 'desconocida')}, "
        f"idioma(s) oficial(es) {idiomas}, moneda {monedas}."
    )

# endregion
