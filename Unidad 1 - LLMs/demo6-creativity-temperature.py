from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

prompt = "Describe en dos frases una novela de ciencia ficción sobre un faro"

for temperature in [0.1, 0.5, 0.9]:
    for i in range(3):
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=50,
            temperature=temperature
        )
    
        print(f"\n--- temperature {temperature} - intento {i + 1} ---")
        print(response.output_text)