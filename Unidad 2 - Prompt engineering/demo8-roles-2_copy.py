from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

prompt1 = "Hola, mi nombre es Francisco"
prompt2 = "Cómo es mi nombre?"

def obtener_respuesta(prompt):
    # Aqui la funcion puede recibir una lista de mensajes y reenviar el
    # historial completo al modelo para conservar el contexto.
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
# Se agrega la respuesta con rol assistant para conservar la alternancia de
# roles y que el segundo turno tenga el contexto de la respuesta anterior.
conversación.append({"role": "assistant", "content": respuesta1.output_text})
conversación.append({"role": "user", "content": prompt2})
# A diferencia de demo8-roles-2.py, aqui se reenvia todo el historial; la
# memoria conversacional la administra este programa cliente.
respuesta2 = obtener_respuesta(conversación)

print("=== Respuesta 2 ===")
print("Respuesta:", respuesta2.output_text)
