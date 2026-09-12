from dotenv import load_dotenv
import os
import json
import requests
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

API_BASE = "http://localhost:3000"
MODELO = "gpt-4.1-nano"

# Lista de respaldo por si la API local no está corriendo
PAISES_FALLBACK = ["Argentina", "Brazil", "Spain"]


# Integración con la API de países

def consultar_api(ruta):
    """
    Hace un GET a la API de países dejando registro en consola.
    """
    url = f"{API_BASE}{ruta}"
    print(f"  -> [API] GET {url}")

    response = requests.get(url, timeout=10)

    print(f"  <- [API] {response.status_code}")

    response.raise_for_status()
    return response.json()


def obtener_catalogo():
    try:
        return consultar_api("/countries")
    except Exception as error:
        print(f"[aviso] No se pudo consultar la API, uso catálogo fijo: {error}")
        return [{"name": pais} for pais in PAISES_FALLBACK]


def obtener_info_pais(nombre):
    """Devuelve los datos detallados de un país, o None si falla la consulta."""
    try:
        return consultar_api(f"/countries/{nombre}")
    except Exception as error:
        print(f"[aviso] No se pudo obtener info de {nombre}: {error}")
        return None


# Paso 1: DETECCIÓN

def detectar_pais(mensaje, paises):
    
    prompt = f"""Sos un clasificador. Tu única tarea es identificar si el mensaje
            del usuario pregunta por alguno de estos países:

            {", ".join(paises)}

            Reglas:
            - Si el mensaje menciona o alude a uno de esos países, devolvé su nombre exacto de la lista.
            - Si menciona un país que NO está en la lista, o no habla de ningún país, devolvé "ninguno".
            - No expliques nada, no respondas la pregunta del usuario.

            Mensaje del usuario: "{mensaje}"
            """

    response = client.responses.create(
        model=MODELO,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "deteccion_pais",
                "schema": {
                    "type": "object",
                    "properties": {
                        "pais": {
                            "type": "string",
                            "enum": paises + ["ninguno"],
                        }
                    },
                    "required": ["pais"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        },
    )

    pais = json.loads(response.output_text)["pais"]
    return None if pais == "ninguno" else pais


# Pasos 2 y 3: INYECCIÓN + RESPUESTA

def responder(mensaje, historial, catalogo, datos_pais=None):
    """
    Genera la respuesta final.

    El system prompt tiene dos bloques:
      1. Instrucciones + catálogo completo -> IDÉNTICO en cada llamada.
      2. Detalle del país detectado (si lo hay) -> distinto en cada turno,
         siempre al final, para no romper el prefijo cacheable del bloque 1.
    """
    system_prompt = f"""Sos un asistente conversacional especializado en información de países.

Catálogo de países disponibles (nombre, capital, región, población):
{json.dumps(catalogo, ensure_ascii=False, indent=2)}

Reglas:
- Usá el catálogo y, si está presente, el detalle del país consultado como
  única fuente de verdad.
- Si el usuario pregunta por un país que no está en el catálogo, aclará que
  no tenés datos verificados de ese país.
- Para preguntas generales que no son sobre países, respondé normalmente.
- Sé breve.
"""

    if datos_pais:
        system_prompt += f"""
Detalle ampliado del país consultado en este turno:
{json.dumps(datos_pais, ensure_ascii=False, indent=2)}
"""

    entrada = [{"role": "system", "content": system_prompt}]
    entrada += historial
    entrada.append({"role": "user", "content": mensaje})

    response = client.responses.create(
        model=MODELO,
        input=entrada,
        max_output_tokens=1000,
    )

    uso = response.usage
    cacheados = getattr(uso.input_tokens_details, "cached_tokens", 0)
    nuevos = uso.input_tokens - cacheados
    porcentaje = (cacheados / uso.input_tokens * 100) if uso.input_tokens else 0
    print(
        f"  [cache] input_tokens={uso.input_tokens} "
        f"cached={cacheados} nuevos={nuevos} ({porcentaje:.0f}% cacheado)"
    )

    return response.output_text


# Flujo de conversación

def main():
    catalogo = obtener_catalogo()
    nombres_paises = [pais["name"] for pais in catalogo]

    print("Chatbot de países. Escribí 'salir' para terminar.")
    print(f"Catálogo cargado: {len(nombres_paises)} países\n")

    historial = []

    while True:
        mensaje = input("Vos: ").strip()

        if not mensaje:
            continue
        if mensaje.lower() in ("salir", "exit", "quit"):
            print("¡Hasta luego!")
            break

        # Paso 1: detección
        pais = detectar_pais(mensaje, nombres_paises)

        # Paso 2: inyección (solo el detalle variable, el catálogo ya está fijo)
        datos_pais = obtener_info_pais(pais) if pais else None
        if pais:
            print(f"[debug] País detectado: {pais}")

        # Paso 3: respuesta
        respuesta = responder(mensaje, historial, catalogo, datos_pais)
        print(f"Bot: {respuesta}\n")

        historial.append({"role": "user", "content": mensaje})
        historial.append({"role": "assistant", "content": respuesta})


if __name__ == "__main__":
    main()
