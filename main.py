import json
from dotenv import load_dotenv
from lib.messages import UserMessage, SystemMessage, ToolMessage # Different message types
from lib.tooling import tool # Tool decorator for creating AI tools
from lib.llm import LLM # Language Model wrapper

load_dotenv()

chat_model = LLM()

# Basic interaction single-turn query
response = chat_model.invoke("What is an AI Agent?")
print("Single Query Response:\n", response.content)
