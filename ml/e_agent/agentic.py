# use gemini
from google import genai
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
client = genai.Client(api_key=API_KEY)
question = "Explain what AI is in one sentence."

def agentic_action(client: genai.Client, question: str):
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=question,
    )
    return response.text