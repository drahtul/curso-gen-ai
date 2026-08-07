from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

prompt = """Describe el producto siguiendo exactamente este formato:

Producto: Cafetera automática
Descripción: Prepara café de forma rápida y sencilla con solo presionar un botón.
Ventaja: Ideal para ahorrar tiempo en las mañanas.

Producto: Mochila impermeable
Descripción: Protege tus pertenencias incluso en condiciones de lluvia intensa.
Ventaja: Perfecta para viajes y uso diario.

Producto: Auriculares inalámbricos con cancelación de ruido
Descripción:
Ventaja:
"""

response = client.responses.create(
        model="gpt-4.1-nano",
        input=prompt,
        max_output_tokens=100
)

print(response.output[0].content[0].text)
