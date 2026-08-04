from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

prompt = "Donde vivirán los humanos en el futuro?"
temp = 1

for i in range(3):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=30,
        temperature=temp
    )
    
    print(f"\nTemperatura {temp}:")
    print(response.output_text)