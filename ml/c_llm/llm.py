# use gemini
from google import genai
import os, time
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
client = genai.Client(api_key=API_KEY)
sample_question = "Explain what AI is in one sentence."

def agentic_action(client: genai.Client, question, name="gemini-2.5-flash-lite", instructions=''):
    response = client.models.generate_content(
        model=name,
        config=genai.types.GenerateContentConfig(
            system_instruction=instructions),
        contents=question,
    )
    return response.text

# alt: gemini-3-flash-preview