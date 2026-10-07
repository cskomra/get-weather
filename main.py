import json
from dotenv import load_dotenv
from lib.messages import UserMessage, SystemMessage, ToolMessage # Different message types
from lib.tooling import tool, Tool # Tool decorator for creating AI tools
from lib.llm import LLM # Language Model wrapper
from typing import cast

load_dotenv()

chat_model = LLM()

# Building an AI Tool
@tool
def get_weather(city: str):
    """Get the current temperature for a city.
    
    Args:
        city (str): Name of the city to check weather for
        
    Returns:
        dict: Contains temperature information for the requested city
    """
    # In a real application, this would call a weather API
    mock_weather = {
        "Sao Paulo": "25°C",
        "Oslo": "-3°C",
        "New York": "10°C",
        "Tokyo": "23°C"
    }
    return {"temperature": mock_weather.get(city, "Unknown")}



# Bind the tool to the LLM
chat_model_with_tools = LLM(tools=[cast(Tool, get_weather)])

# Tool usage flow
messages = [
    SystemMessage(content="You are a helpful assistant that can access a tool to get current temperature "
                  "for cities. Use the tool whenever someone asks about the weather or temperature "
                  "in a specific location. Inform the user if you don't know the answer."
                  ),
    UserMessage(content="How cold is it in Oslo?")
]

# AI decides to use the weather tool
ai_message = chat_model_with_tools.invoke(messages)
print(ai_message)

# Check messages structure
messages.append(ai_message)
print("Updated messages structure:\n", messages)

# Tool call id will be required later when creating the ToolMessage
tool_call_id = messages[-1].tool_calls[0].id
print("Tool call ID:\n", tool_call_id)

# Extract the arguments
args = json.loads(messages[-1].tool_calls[0].function.arguments)
print("Extracted arguments:\n", args)

# Execute the tool with the AI's requested parameters
tool_result = get_weather(**args)
print("Tool execution result:\n", tool_result)

# Create a tool response message
tool_message = ToolMessage(
    tool_call_id=tool_call_id,
    name="get_weather",
    content=json.dumps(tool_result),
)
print("Tool message:\n", tool_message)

# Check messages structure
messages.append(tool_message)
print("Updated messages structure after tool message:\n", messages)

# Let AI formulate final response
ai_message = chat_model_with_tools.invoke(messages)
print("Final AI message:\n:", ai_message)

# Check messages structure
messages.append(ai_message)
print("Updated messages structure after final AI message:\n", messages)