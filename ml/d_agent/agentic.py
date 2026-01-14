from llm import query
from google import genai
import subprocess
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
client = genai.Client(api_key=os.getenv('API_KEY'))

def generate_applescript(client: genai.Client, input_command: str):
    intput_command = input_command.strip().lower()
    instructions = (
        "You are a natural-language-to-AppleScript translator. "
        "Your output must be 100% executable AppleScript code. "
        "Strict Rule: Do not include explanations, comments, or markdown code blocks (```). "
        "Strict Rule: Do not include introductory text or follow-up sentences."
        )
    question = (
        f"Translate the following request into raw AppleScript code: '{input_command}'. "
        "Output ONLY the code."
        )
    return query(client, question, instructions)

def execute_command(script): 
    result = subprocess.run(["osascript", "-e", script])
    return result
    

sample = ('open my mail and draft a new email that says hello how are you.' 
          'email subject should be upcoming vacation. email address should be b.de at gmail dot com'
    )