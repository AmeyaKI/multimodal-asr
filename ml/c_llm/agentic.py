from ..c_llm.llm import query
from google import genai


def generate_applescript(client: genai.Client, input_command: str):
    question = f"Plan and generate only the necessary AppleScript code for this task: {input_command}"
    instructions = "You are a smart assistant that breaks down complex tasks into efficient AppleScript commands"
    return query(client, question, instructions)