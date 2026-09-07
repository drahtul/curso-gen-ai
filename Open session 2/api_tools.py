import requests
from langchain_core.tools import tool

# API de países de la Open session 1 (node paises-api.js)
COUNTRIES_API_URL = "http://localhost:3000"

# API pública de clima
OPEN_METEO_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HTTP_TIMEOUT = 15

# Códigos WMO que devuelve Open-Meteo en `weathercode`.
WEATHER_CODES = {
    0: "despejado",
    1: "mayormente despejado",
    2: "parcialmente nublado",
    3: "nublado",
    45: "niebla",
    48: "niebla con escarcha",
    51: "llovizna leve",
    53: "llovizna moderada",
    55: "llovizna intensa",
    56: "llovizna helada leve",
    57: "llovizna helada intensa",
    61: "lluvia leve",
    63: "lluvia moderada",
    65: "lluvia intensa",
    66: "lluvia helada leve",
    67: "lluvia helada intensa",
    71: "nevada leve",
    73: "nevada moderada",
    75: "nevada intensa",
    77: "granos de nieve",
    80: "chubascos leves",
    81: "chubascos moderados",
    82: "chubascos violentos",
    85: "chubascos de nieve leves",
    86: "chubascos de nieve intensos",
    95: "tormenta eléctrica",
    96: "tormenta con granizo leve",
    99: "tormenta con granizo intenso",
}


@tool
def consultar_clima(ciudad: str) -> str:
    """Consulta el clima ACTUAL y el pronóstico de hoy de una ciudad (Open-Meteo).

    Devuelve temperatura, sensación de condición, viento, máxima/mínima del día
    y probabilidad de lluvia. Recibe el nombre de una ciudad, por ejemplo
    'Montevideo' o 'Madrid'. Si el usuario pregunta por la capital de un país,
    primero obtené la capital con `consultar_pais` y luego llamá a esta tool.
    """
    try:
        geo = requests.get(
            OPEN_METEO_GEO_URL,
            params={"name": ciudad, "count": 1, "language": "es"},
            timeout=HTTP_TIMEOUT,
        ).json()
    except requests.RequestException as exc:
        return f"ERROR: no se pudo contactar el servicio de geocodificación ({exc})."

    resultados = geo.get("results") or []
    if not resultados:
        return (
            f"SIN_RESULTADOS: no se encontraron coordenadas para '{ciudad}', "
            "por lo que no hay datos de clima disponibles."
        )

    lugar = resultados[0]
    nombre = ", ".join(
        p for p in (lugar.get("name"), lugar.get("admin1"), lugar.get("country")) if p
    )

    try:
        clima = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": lugar["latitude"],
                "longitude": lugar["longitude"],
                "current_weather": True,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=HTTP_TIMEOUT,
        ).json()
    except requests.RequestException as exc:
        return f"ERROR: no se pudo obtener el clima de {nombre} ({exc})."

    actual = clima.get("current_weather")
    if not actual:
        return f"SIN_RESULTADOS: no hay datos de clima disponibles para {nombre}."

    condicion = WEATHER_CODES.get(actual.get("weathercode"), "condición desconocida")
    lineas = [
        f"Clima en {nombre}:",
        f"- Temperatura actual: {actual['temperature']} °C ({condicion})",
        f"- Viento: {actual['windspeed']} km/h",
    ]

    diario = clima.get("daily") or {}
    if diario.get("time"):
        lineas.append(
            f"- Hoy ({diario['time'][0]}): máxima {diario['temperature_2m_max'][0]} °C, "
            f"mínima {diario['temperature_2m_min'][0]} °C"
        )
        prob = diario.get("precipitation_probability_max", [None])[0]
        lluvia = diario.get("precipitation_sum", [None])[0]
        if prob is not None:
            lineas.append(f"- Probabilidad de precipitación hoy: {prob}%")
        if lluvia is not None:
            lineas.append(f"- Precipitación acumulada esperada: {lluvia} mm")

    return "\n".join(lineas)


@tool
def consultar_pais(pais: str) -> str:
    """Consulta datos de un PAÍS: capital, moneda, población, idiomas, región, etc.

    El nombre del país debe pasarse en INGLÉS (por ejemplo 'Japan', 'Brazil',
    'Norway', 'France'), tal como lo espera la API de países.
    """
    url = f"{COUNTRIES_API_URL}/countries/{requests.utils.quote(pais)}"

    try:
        respuesta = requests.get(url, timeout=HTTP_TIMEOUT)
    except requests.RequestException:
        return (
            "ERROR: la API de países no está disponible. "
            f"Verificá que esté corriendo en {COUNTRIES_API_URL} "
            "(`npm start` dentro de 'Open session 1')."
        )

    if respuesta.status_code == 404:
        return f"SIN_RESULTADOS: no se encontró información del país '{pais}'."
    if not respuesta.ok:
        return f"ERROR: la API de países respondió {respuesta.status_code}."

    d = respuesta.json()
    monedas = ", ".join(
        f"{m.get('name')} ({m.get('code')} {m.get('symbol', '')})".strip()
        for m in d.get("currencies", [])
    ) or "desconocida"

    return "\n".join(
        [
            f"País: {d.get('name')} ({d.get('officialName')})",
            f"- Capital: {d.get('capital')}",
            f"- Región: {d.get('region')} / {d.get('subregion')}",
            f"- Población: {d.get('population')}",
            f"- Superficie: {d.get('areaKm2')} km2",
            f"- Idiomas oficiales: {', '.join(d.get('languages', [])) or 'desconocidos'}",
            f"- Moneda: {monedas}",
            f"- Zonas horarias: {', '.join(d.get('timezones', [])) or 'desconocidas'}",
            f"- Gentilicio: {d.get('demonym')}",
        ]
    )
