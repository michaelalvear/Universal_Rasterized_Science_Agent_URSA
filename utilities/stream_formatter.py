"""
This function formats the agent's stream output
"""
from langchain_core.messages import BaseMessage
import json


def format_stream(message: BaseMessage) -> str:
    """
    :param message:
    :return:
    """
    header = f"================================ {message.type.upper()} ================================"

    # If message content is a normal string, return it
    if isinstance(message.content, str):
        content = message.content
    # If it's a Gemini list, extract the .text attribute
    else:
        content = getattr(message, 'text', str(message.content))

    # Extract Tool Calls (If the AI is requesting a tool)
    tool_info = ""
    if hasattr(message, "tool_calls") and message.tool_calls:
        for call in message.tool_calls:
            # Accessing the 'name' and id' keys from the tool call dictionary
            tool_info += f"\n[TOOL NAME]: {call['name']}"
            tool_info += f"\n[TOOL CALL ID]: {call['id']}"

            # Format the arguments nicely as JSON
            args = json.dumps(call['args'], indent=2)
            tool_info += f"\n[ARGS]: {args}\n"

    # Extract Tool Results (If it's a ToolMessage)
    if message.type == "tool":
        tool_info += f"\n[RESPONDING TO ID]: {message.tool_call_id}\n"

    # Combine everything
    final_content = content + tool_info

    tail = '=' * len(header)

    formatted_string = header + '\n' + final_content + '\n' + tail

    return formatted_string
