from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

user_input = input("Qué deseas saber: ")

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

response_text = response.output[0].content[0].text

print(response.output[0].content[0].text)
