from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Una instruccion explicita orienta al modelo hacia autocompletar el contexto,
# en lugar de interpretarlo como una pregunta conversacional.
instrucciones = "Continua el texto del usuario. No respondas, no expliques. Responde unicamente con UNA palabra."

# prompt = "El gato esta en el"
#sin instruct
prompt = "El gato esta en el patio trasero"

response = client.responses.create(
    model="gpt-4o-mini",
    instructions=instrucciones,
    input=prompt,
    # El limite se mide en tokens, no en palabras; una palabra puede ocupar
    # varios tokens y la instruccion no garantiza por si sola una sola palabra.
    max_output_tokens=16
)

# El prompt termina en un contexto parcialmente abierto para que el modelo
# prediga una continuacion probable, no para que repita una respuesta fija.
# primera_palabra = response.output_text.strip().split()[0] if response.output_text.strip() else ""
print("Prompt:", prompt)
print("Continuacion:", response.output_text)
