import os
import time
import builtins
import uuid
from typing import Any, Dict, List
from openai import AsyncOpenAI
import tiktoken

# Record start time
start_time = time.time()

# Save the original print function for timed logging
original_print = builtins.print

# Create a custom print function that shows elapsed time
def timed_print(*args, **kwargs):
    elapsed = time.time() - start_time
    original_print(f"[{elapsed:.3f}s]", *args, **kwargs)

# Replace the built-in print function
builtins.print = timed_print

COLORS = {'GREEN': '\033[32m', 'RESET': '\033[0m'}

# Helper function to safely decode bytes or return strings as is
def safe_decode(data):
    """Safely decode bytes or return the string as is."""
    if isinstance(data, bytes):
        return data.decode('utf-8', errors='replace')
    return data

# OpenAI utility functions using the async client
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def count_tokens(messages: list[dict[str, Any]], model: str = "gpt-4") -> int:
    """Count the number of tokens in a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    
    for message in messages:
        # Every message follows format: {"role": role, "content": content}
        num_tokens += 4  # Every message follows format: <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
    
    return num_tokens

def trim_messages(messages: list[dict[str, Any]], max_tokens: int, model: str = "gpt-4") -> list[dict[str, Any]]:
    """Trim messages to fit within token limit by removing older messages."""
    while len(messages) > 1 and count_tokens(messages, model) > max_tokens:
        messages.pop(1)
    return messages

async def get_model_response(messages: list[dict[str, Any]]) -> str:
    messages = trim_messages(messages, 110000)
    
    chat_completion = await client.chat.completions.create(
        messages=messages, # type: ignore
        model="gpt-4o",
    )
    return chat_completion.choices[0].message.content # type: ignore

# Execution result class compatible with the interface
class ExecutionResult:
    """Compatible ExecutionResult class for ComputerInterface"""
    def __init__(self, output: bytes, exit_code: int):
        self.output = output
        self.exit_code = exit_code