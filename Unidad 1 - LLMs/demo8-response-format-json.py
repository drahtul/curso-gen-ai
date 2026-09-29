from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# El esquema restringe la forma de la salida durante la generacion. Esto es
# mas fiable que pedir "responde en JSON" solamente mediante el prompt.
response = client.responses.create(
    model="gpt-4o-mini",
    input="Dame un estudiante en JSON",
    text={
        "format": {
            "type": "json_schema",
            "name": "estudiante",
            "schema": {
                "type": "object",
                "properties": {
                    "nombre": {"type": "string"},
                    "edad": {"type": "integer"}
                },
                "required": ["nombre", "edad"],
                # Impide que el modelo agregue propiedades fuera del contrato.
                "additionalProperties": False
            },
            # strict exige respetar el esquema definido por la aplicacion.
            "strict": True
        }
    }
)

# Sigue siendo texto JSON; para usarlo como estructura Python habria que
# analizarlo con un parser JSON.
print(response.output_text)