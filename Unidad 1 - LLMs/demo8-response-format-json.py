from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

response = client.responses.create(
    model="gpt-4o-mini",
    temperature=1.0,
    top_p=1.0,
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
                "additionalProperties": False
            },
            "strict": True
        }
    }
)

print(response.output_text)