# use gemini
from google import genai
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
client = genai.Client(api_key=API_KEY)
sample_question = "Explain what AI is in one sentence."

def query(client: genai.Client, question: str, instructions: str, name: str = "gemini-2.5-flash-lite"):
    """Querying Gemini model

    Args:
        client (genai.Client): initializing Gemini Client class
        question (str): user query to model
        instructions (str): custom default instructions for model
        name (str, optional): name of Gemini model. Defaults to "gemini-2.5-flash-lite" for latency optimization.

    Returns:
        text(str): LLM text response to question
    """
    response = client.models.generate_content(
        model=name,
        config=genai.types.GenerateContentConfig(
            system_instruction=instructions
        ),
        contents=question,
    )
    return response.text

# alt: gemini-3-flash-preview