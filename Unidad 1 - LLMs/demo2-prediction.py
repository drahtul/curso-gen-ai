from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Explicar por qué el modelo no puede continuar el texto de manera coherente si no se le da una instrucción clara.
instrucciones = "Continua el texto del usuario. No respondas, no expliques. Responde unicamente con UNA palabra."

prompt = "El gato esta en el"

response = client.responses.create(
    model="gpt-4o-mini",
    instructions=instrucciones,
    input=prompt,
    max_output_tokens=16
)

# primera_palabra = response.output_text.strip().split()[0] if response.output_text.strip() else ""
print("Prompt:", prompt)
print("Continuacion:", response.output_text)
