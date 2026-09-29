from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# La instruccion pide descomponer el problema en pasos intermedios antes de
# concluir, una estrategia de prompting para hacer explicito el procedimiento.
prompt = """¿Conviene más comprar un producto de $100 con 20% de descuento o uno de $80 sin descuento?
Analiza paso a paso antes de responder."""

response = client.responses.create(
        # El razonamiento generado es una explicacion del modelo, no una
        # prueba formal: el calculo debe verificarse de manera independiente.
        model="gpt-4.1-nano",
        input=prompt,
        max_output_tokens=1000
)

print(response.output[0].content[0].text)
