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
    response = client.responses.create(
        model="gpt-4.1-nano",
        input=prompt,
        max_output_tokens=100
    )
    return response

# Primer turno (usuario)
conversación = [{"role": "user", "content": prompt1}]
respuesta1 = obtener_respuesta(conversación)

print("=== Respuesta 1 ===")
print("Respuesta:", respuesta1.output_text)

# Segundo turno (incluye historial: user -> assistant -> user)
conversación.append({"role": "assistant", "content": respuesta1.output_text})
conversación.append({"role": "user", "content": prompt2})
respuesta2 = obtener_respuesta(conversación)

print("=== Respuesta 2 ===")
print("Respuesta:", respuesta2.output_text)
