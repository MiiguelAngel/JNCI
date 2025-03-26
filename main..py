import openai
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def test_api():
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "¿Puedes confirmar que esta prueba de la API funciona correctamente?"}
        ]
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    test_api()