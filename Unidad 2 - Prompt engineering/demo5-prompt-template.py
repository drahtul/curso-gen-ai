from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Estos valores son variables de una plantilla reutilizable: permiten cambiar
# la tarea sin reescribir toda la instruccion fija.
tema = input("Tema: ")
audiencia = input("Audiencia: ")
formato = input("Formato: ")
detalle = input("Detalle: ")

# El f-string combina instrucciones estables con datos de tiempo de ejecucion.
# Los valores no se validan ni normalizan antes de enviarse al modelo.
prompt = f"""Explica el concepto de {tema}
para {audiencia}
en formato {formato}
incluyendo {detalle}
"""

response = client.responses.create(
        # La plantilla controla el contexto, pero no garantiza que el modelo
        # respete perfectamente el formato pedido.
        model="gpt-4.1-nano",
        input=prompt,
        max_output_tokens=1000
)

print(response.output[0].content[0].text)
