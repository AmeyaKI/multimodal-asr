import subprocess
from langchain.tools import tool
import os
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
# https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('API_KEY') # type: ignore

model = init_chat_model('google_genai:gemini-2.5-flash-lite')

@tool
def execute_applescript(script: str) -> str:
    """
    Internally executes AppleScript on macOS 
    Returns stdout or error.
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            output = result.stdout.strip() if result.stdout else "Command executed successfully"
            return f"SUCCESS: {output}"
        else:
            error = result.stderr.strip() if result.stderr else "Unknown error"
            return f"FAILED: {error}"
    except subprocess.TimeoutExpired:
        return "FAILED: AppleScript execution timed out"
    except Exception as e:
        return f"FAILED: {str(e)}"
    
@tool
def mac_tts(text: str) -> str:
    """Convert text to speech using macOS 'say' command."""
    try:
        subprocess.run(["say", text], check=True, timeout=30)
        return f"Spoke: {text}"
    except subprocess.TimeoutExpired:
        return "Speech timed out"
    except Exception as e:
        return f"Speech error: {str(e)}"
    
APPLESCRIPT_AGENT_INSTRUCTIONS = """You are an AppleScript code generator.
Convert natural language commands into executable AppleScript code.

Rules:
- Return ONLY the AppleScript code, no explanations or markdown
- No code blocks or backticks
- Make the code as simple and direct as possible
- Handle common macOS applications like Safari, Finder, Messages, Mail, etc.

Examples:
Command: "Open Safari"
AppleScript: tell application "Safari" to activate

Command: "Show me the current time"
AppleScript: display dialog (current date) as string

Command: "Set volume to 50%"
AppleScript: set volume output volume 50
"""

# create agent
applescript_agent = create_agent(
    model,
    tools=[execute_applescript],
    system_prompt=APPLESCRIPT_AGENT_INSTRUCTIONS,
)

def test_applescript_agent(command: str):
    """Test the agent with a natural language command"""
    print(f"\n🎤 Testing command: {command}\n")
    
    # Invoke the agent
    result = applescript_agent.invoke({
        "messages": [{"role": "user", "content": f"Execute this command: {command}. Generate and run the AppleScript."}]
    })
    
    # Print all messages to see the flow
    print("="*60)
    for message in result["messages"]:
        print(f"\n{message.type.upper()}:")
        if hasattr(message, 'content'):
            print(message.content)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print(f"Tool calls: {message.tool_calls}")
    print("="*60)

