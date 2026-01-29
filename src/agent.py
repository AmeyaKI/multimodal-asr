# Libraries
import subprocess
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

# relevant files
from tts import TTS
tts_model = TTS() # load TTS (Default: Inworld)

# API key setup
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('API_KEY') #  type: ignore
model = init_chat_model('google_genai:gemini-2.5-flash-lite')


### AppleScript Agent ###
APPLESCRIPT_INSTRUCTIONS="""
You are a natural-language-to-AppleScript translator. 
Your output must be 100% executable AppleScript code. 
Strict Rule: Do not include explanations, comments, or markdown code blocks.
Strict Rule: Do not include introductory text or follow-up sentences
Strict Rule: Include proper spacing and indentation coupled with the latest fully functional AppleScript syntax.
"""





applescript_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=APPLESCRIPT_INSTRUCTIONS,
    checkpointer=InMemorySaver(),
)
@tool
def execute_applescript(script: str): 
    result = subprocess.run(["osascript", "-e", script])
    return result

def text_to_speech(text: str):
    try:
        tts_model.speak(text)
    except Exception as e:
        return f"Speech error: {e}"

### Supervisor Agent ###

SUPERVISOR_INSTRUCTIONS = """You are a helpful voice assistant for macOS.

CRITICAL WORKFLOW - Follow this EXACT sequence:

1. FIRST: Immediately respond with a quick, natural acknowledgement
   - Use casual responses like: "On it", "Sure thing", "Got it", "Of course", "Right away"
   - Keep it SHORT (1-3 words maximum)
   - This shows you're listening and working on it

2. THEN: Use the run_applescript_command tool with the EXACT user request
   - Pass the user's original command DIRECTLY to the tool
   - Do NOT rephrase or modify the user's request
   - Example: If user says "Open Safari", pass exactly "Open Safari" to the tool

3. FINALLY: After getting the tool result, provide a brief confirmation
   - If SUCCESS: "Done", "All set", "Opened Safari", etc.
   - If FAILED: "Couldn't do that" or explain the error briefly

IMPORTANT:
- Always acknowledge BEFORE using tools
- Pass user's EXACT request to run_applescript_command (don't modify it)
- Keep all responses conversational and brief (they will be spoken aloud)
- Don't be verbose or over-explain
- Match the casual tone of the user

Example flow:
User: "Open Safari"
You: "On it" 
[call run_applescript_command with "Open Safari"]
[receive result]
You: "Opened"
"""

@tool
def execute_applescript_agent(command: str):
    """Execute macOS command by calling applescript sub-agent

    Args:
        command (str): user command
        
    """
    result = applescript_agent.invoke({
        "messages": [{"role": "user", "content": "command"}]
    })
    return result['messages'][-1].content

supervisor_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=SUPERVISOR_INSTRUCTIONS,
    checkpointer=InMemorySaver(),
)