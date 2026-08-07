from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

tema = input("Tema: ")
audiencia = input("Audiencia: ")
formato = input("Formato: ")
detalle = input("Detalle: ")

prompt = f"""Explica el concepto de {tema}
para {audiencia}
en formato {formato}
incluyendo {detalle}
"""

response = client.responses.create(
        model="gpt-4.1-nano",
        input=prompt,
        max_output_tokens=1000
)

print(response.output[0].content[0].text)
