import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # o usa directamente tu clave

models = openai.models.list()

for model in models.data:
    print(model.id)