from dotenv import load_dotenv
import os
from openai import OpenAI
import json

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# La lista de mensajes separa instrucciones de comportamiento (system) del
# pedido concreto del usuario (user), una distincion central de los roles.
conversation = [
    {"role": "system", "content": "Eres un profesor de bases de datos claro y didáctico. Solo respondes en formato JSON."},
    {"role": "user", "content": "Explica qué es una clave primaria."}
]

response = client.responses.create(
    # El modelo interpreta cada mensaje segun su rol dentro de la misma
    # solicitud y recibe ambos elementos como contexto.
        model="gpt-4.1-nano",
        input=conversation,
        max_output_tokens=300
)

print(json.dumps(response.model_dump(), indent=4))
