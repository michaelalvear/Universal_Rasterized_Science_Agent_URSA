"""
This function formats the agent's stream output
"""
from langchain_core.messages import BaseMessage, ToolMessage
import json


def format_msg(msg: BaseMessage) -> str:
    """
    Formats the agent's stream output.
    """
    header = f"{'=' * 32} {msg.type.upper()} {'=' * 32}"

    # Handle the formatting of non tool messages
    content = ""
    if not isinstance(msg, ToolMessage):
        text = getattr(msg, 'text', str(msg.content))
        if text:
            content += f"\n{text}"

    # Extract Tool Calls from message (if available)
    tool_calls = ""
    if hasattr(msg, "tool_calls") and msg.tool_calls:

        tool_calls += f"\n\n{'+' * 16} TOOL CALLS {'+' * 16}"
        for call in msg.tool_calls:
            # Accessing the 'name' and id' keys from the tool call dictionary
            tool_calls += f"\n[TOOL NAME]: {call['name']}"
            tool_calls += f"\n[TOOL CALL ID]: {call['id']}"

            # Format the arguments nicely as JSON
            args = json.dumps(call['args'], indent=2)
            tool_calls += f"\n[ARGS]: {args}"
        tool_calls += f"\n{'+' * 44}\n"

    # Extract Tool Results (If it's a ToolMessage)
    tool_results = ""
    if isinstance(msg, ToolMessage):
        tool_results += f"\n[TOOL NAME]: {msg.name}"
        tool_results += f"\n[Content]: \n{msg.content}"
        tool_results += f"\n[RESPONDING TO ID]: {msg.tool_call_id}"

    # Combine everything
    final_content = content + tool_calls + tool_results

    tail = f"\n{'=' * len(header)}"

    formatted_string = header + final_content + tail

    return formatted_string
