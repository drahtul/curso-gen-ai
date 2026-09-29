from dotenv import load_dotenv
import os
from openai import OpenAI
import json

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

prompt1 = "Hola, mi nombre es Francisco"
prompt2 = "Cómo es mi nombre?"

def obtener_respuesta(prompt):
    # La funcion encapsula una llamada, pero no conserva estado por si sola:
    # cada solicitud recibe solo un mensaje de usuario.
    response = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {"role": "user", "content": prompt}
        ],
        max_output_tokens=100
    )

    return response

respuesta1 = obtener_respuesta(prompt1)
# Esta segunda llamada es independiente: no recibe prompt1 ni respuesta1.
respuesta2 = obtener_respuesta(prompt2)

print("=== Respuesta 1 ===")
print("Respuesta:", respuesta1.output_text)

print("=== Respuesta 2 ===")
print("Respuesta:", respuesta2.output_text)
