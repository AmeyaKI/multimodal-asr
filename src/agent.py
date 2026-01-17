# Libraries
import subprocess
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

# relevant files
from tts import TTS

# API Key
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('API_KEY') #  type: ignore

# model = init_chat_model('google_genai:gemini-2.5-flash-lite')

@tool
def execute_command(script): 
    subprocess.run(["osascript", "-e", script])

@tool
def speak(script):
    ...
    
