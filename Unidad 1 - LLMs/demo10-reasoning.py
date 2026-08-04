from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

prompt = "Un bate y una pelota cuestan $1.10 en total. " \
         "El bate cuesta $1.00 más que la pelota. " \
         "¿Cuánto cuesta la pelota?"

# El reasoning effort puede ser: 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', and 'max'.

for effort in ["low", "xhigh"]:
    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
        reasoning={"effort": effort}
    )

    print(f"\n--- Reasoning effort: {effort} ---")
    print("Respuesta:", response.output_text)
    print("Tokens de input:", response.usage.input_tokens)
    print("Tokens de output:", response.usage.output_tokens)
    print("Tokens de razonamiento:", response.usage.output_tokens_details.reasoning_tokens)
    print("Tokens totales:", response.usage.total_tokens)
