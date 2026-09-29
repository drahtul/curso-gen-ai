from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# La Responses API recibe un contexto de entrada y genera texto condicionado
# por ese contexto; no busca una respuesta fija en una base de datos.
response = client.responses.create(
    model="gpt-4o-mini",
    input="Hola mundo!"
)

# output_text es una vista comoda del texto generado. La respuesta completa
# tambien contiene metadatos utiles, como el uso de tokens.
print(response.output_text)