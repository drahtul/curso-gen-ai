from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

prompt = "El futuro de la inteligencia artificial será"

for top_p in [0.1, 0.5, 0.9, 1.0]:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=50,
        temperature=1.0,
        top_p=top_p
    )
    
    print(f"\n--- top_p {top_p} ---")
    print(response.output_text)