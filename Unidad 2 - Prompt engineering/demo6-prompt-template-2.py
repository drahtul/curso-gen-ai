from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# La entrada variable permite reutilizar el mismo patron para tareas distintas
# sin cambiar la estructura fija de la instruccion.
user_input = input("Qué deseas saber: ")

# Se separa la instruccion fija del contenido del usuario mediante una
# plantilla. Pedir pasos intermedios busca facilitar la comprension y revision.
prompt = f"""Resuelve la siguiente tarea paso a paso:

{user_input}

Primero explica tu razonamiento.
Luego da la respuesta final de forma clara.
"""

response = client.responses.create(
        model="gpt-4.1-nano",
        input=prompt,
        max_output_tokens=1000
)

# Se extrae el texto para mostrar como obtenerlo de la respuesta. La variable
# se conserva tal como esta definida en el ejemplo.
response_text = response.output[0].content[0].text

print(response.output[0].content[0].text)
