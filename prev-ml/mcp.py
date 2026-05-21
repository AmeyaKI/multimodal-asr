import asyncio
import json
import os
from pathlib import Path


from google import genai 
from mcp import Client

# -----------------------------
# CONFIG
# -----------------------------

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("Set API_KEY environment variable")

MODEL_NAME = "gemini-2.5-flash-lite"

# Absolute path to applescript-mcp build output
MCP_SERVER_PATH = f"{str(Path.cwd())}/../applescript-mcp/dist/index.js"

client = genai.Client(api_key=API_KEY)


# -----------------------------
# MCP CLIENT
# -----------------------------

class AppleScriptMCP:
    """
    Only layer allowed to talk to the MCP server.
    """

    def __init__(self):
        self.client = Client(
            command="node",
            args=[MCP_SERVER_PATH],
        )

    async def connect(self):
        await self.client.connect()

    async def list_tools(self):
        return await self.client.list_tools()

    async def call(self, tool_name, arguments):
        return await self.client.call_tool(tool_name, arguments)


# -----------------------------
# MCP → GEMINI TOOL CONVERSION
# -----------------------------

def convert_mcp_tools(mcp_tools):
    """
    Convert MCP tools into Gemini function declarations.
    """
    functions = []

    for tool in mcp_tools:
        functions.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        })

    return functions


# -----------------------------
# GEMINI QUERY (MODERN API)
# -----------------------------

def query(
    client: genai.Client,
    contents,
    instructions: str,
    tools=None,
    name: str = MODEL_NAME,
):
    """
    Modern Gemini call with optional tool calling.
    """
    response = client.models.generate_content(
        model=name,
        contents=contents,
        config=genai.types.GenerateContentConfig(
            system_instruction=instructions,
            tools=[{"function_declarations": tools}] if tools else None,
            tool_config=genai.types.ToolConfig(
                function_calling_config=genai.types.FunctionCallingConfig(mode=genai.types.FunctionCallingConfigMode.AUTO)
            ) if tools else None,
        ),
    )
    return response


# -----------------------------
# AGENT LOOP
# -----------------------------

async def run_assistant(user_input: str):
    system_prompt = (
        "You are a macOS personal assistant. "
        "If an action is required, select the correct tool. "
        "Do not explain internal reasoning."
    )

    # Gemini message format
    contents = [
        {"role": "user", "parts": [user_input]}
    ]

    # Connect to MCP
    mcp = AppleScriptMCP()
    await mcp.connect()

    # Load tools
    mcp_tools = await mcp.list_tools()
    gemini_tools = convert_mcp_tools(mcp_tools)

    # First LLM call
    response = query(
        client=client,
        contents=contents,
        instructions=system_prompt,
        tools=gemini_tools,
    )

    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if candidate.content is not None and hasattr(candidate.content, "parts"):
            parts = candidate.content.parts
        else:
            parts = []

        # Check for tool call
        for part in parts:
            fn = getattr(part, "function_call", None)
            if fn is not None:
                tool_name = fn["name"]
                args = fn.get("args", {})

                # Execute MCP tool
                result = await mcp.call(tool_name, args)

                # Feed tool result back to Gemini
                contents.append({
                    "role": "model",
                    "parts": [part],
                })
                contents.append({
                    "role": "tool",
                    "parts": [json.dumps(result)],
                })

                final_response = query(
                    client=client,
                    contents=contents,
                    instructions=system_prompt,
                    tools=gemini_tools,
                )

                return final_response.text

        # No tool call → normal response
        return response.text
    else:
        return "No response candidates returned from Gemini."


# -----------------------------
# CLI ENTRYPOINT
# -----------------------------

if __name__ == "__main__":
    user_text = input("Command: ")
    output = asyncio.run(run_assistant(user_text))
    print("\nAssistant:", output)
