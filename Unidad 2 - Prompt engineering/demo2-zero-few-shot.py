from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

prompt = """Clasifica el sentimiento del siguiente texto como positivo, negativo o neutro:

Texto: "La película fue increíble, me encantó"
Sentimiento: Positivo

Texto: "No me gustó el producto, llegó roto"
Sentimiento: Negativo

Texto: "El lugar está bien, nada especial"
Sentimiento: Neutro

Texto: "El servicio fue lento y la comida estaba fría"
Sentimiento:

"""

response = client.responses.create(
        model="gpt-4.1-nano",
        input=prompt,
        max_output_tokens=100
)

print(response.output[0].content[0].text)
