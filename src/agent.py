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

model = init_chat_model('google_genai:gemini-2.5-flash-lite')

instructions="""
You are a natural-language-to-AppleScript translator. 
Your output must be 100% executable AppleScript code. 
Strict Rule: Do not include explanations, comments, or markdown code blocks.
Strict Rule: Do not include introductory text or follow-up sentences
Strict Rule: Include proper spacing and indentation coupled with the latest fully functional AppleScript syntax.
"""

agent = create_agent(
    model=model,
    system_prompt=instructions,
    checkpointer = InMemorySaver(),
)