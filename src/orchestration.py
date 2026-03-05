"""
This is the main agent orchestration file, run this to talk to the agent in the
console
"""

# For paths and environment variables
import os
import sys
from dotenv import load_dotenv

# URSA modules
from tools import *
from schemas import *

sys.path.append(
    os.path.abspath("../utilities"))  # Add "utilities" dir to the search path
from stream_formatter import format_stream

# Langgraph/Langchain
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode

# Xarray
import xarray as xr

# Types
from typing import Literal, List

load_dotenv()

tools = [
    bisect_context_retriever,
    dataset_metadata_retriever,
    spatial_temporal_select,
    filter_by_value,
    resample_time_series,
    reduce_dimension,
    inspect_selection,
    geocoding_tool
]


# ++++++++++ Graph setup ++++++++++
# Nodes (Some of the node code inspired by):
# https://github.com/langchain-ai/how_to_fix_your_context/blob/main/notebooks/01-rag.ipynb)
def user_input(state: AgentState) -> dict[str, List[BaseMessage]]:
    """
    Append user prompt to state
    """
    user_request = input("~$ ")

    return {"messages": [HumanMessage(content=user_request)]}


def end_session_router(
        state: AgentState
) -> Literal["session ended", "request created"]:
    """Decide whether to end session"""
    # Was the last human message exit?
    if state.messages[-1].content == "exit":
        return "session ended"
    else:
        return "request created"


# Initialize Gemini API + bind tools
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0,
                             streaming=True).bind_tools(tools)


# Main llm invocation
def llm_call(state: AgentState) -> dict[str, List[BaseMessage]]:
    """
    LLM decides whether to call a tool or not.
    """

    llm_response = llm.invoke(state.messages)
    return {"messages": [llm_response]}


# llm to tool node routing function
def tool_router(state: AgentState) -> Literal["pending tool calls", "done"]:
    """
    Decide if we should continue the tool loop or stop based on whether
    the LLM made a tool call.
    """
    last_message = state.messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "pending tool calls"
    # Otherwise, we stop (return to llm_call)
    return "done"


graph = StateGraph(AgentState)

graph.add_node("user input", user_input)
graph.add_node("tool node", ursa_tool_node)
graph.add_node("llm call", llm_call)

# Edges
graph.add_edge(START, "user input")

graph.add_conditional_edges(
    "user input",
    end_session_router,
    {
        "session ended": END,
        "request created": "llm call"
    }
)

graph.add_conditional_edges(
    "llm call",
    tool_router,
    {
        "pending tool calls": "tool node",
        "done": "user input"
    }
)

graph.add_edge("tool node", "llm call")

app = graph.compile()

# ++++++++++ Initializing Agent ++++++++++

# Initialize state variables
essential_context = """
You are a helpful assistant tasked with retrieving and interpreting information 
from a South Florida hydrological model known as:
Biscayne and Southern Everglades Coastal Transport Model (BISECT).
 
Technical details about the model are recorded in the paper:
"The Hydrologic System of the South Florida Peninsula: 
Development and Application of the Biscayne and Southern 
Everglades Coastal Transport (BISECT) Model"

 Authored by:
*Eric D. Swain, Melinda A. Lohmann, and Carl R. Goodwin*.

Your goal is to make this invaluable knowledge accessible 
to non technical South Florida stakeholders (city council-people, engineers,
developers, etc.).

You have been provided tools to fetch context from the paper itself as well
as a small subset of the results of the model in the form of raster GIS data
tracking surface salinity measurements of a baseline emissions scenario in 
South Florida.

Your can get context on the paper through the tools provided you.

Reflect on any context you fetch, and keep retrieving until you have sufficient 
context to answer the user's research request.

*ALWAYS PROVIDE COMPLETE CITATIONS*

You can also generate subset of the raster data called using the GIS tool suite
provided too you. Follow argument schemas *EXACTLY*. 

The updated data after each operation is preserved in your state, so if you 
need to perform a multistep operation you can.

To see the a statistical summary of the extracted data held in your active 
selection use the inspect_selection tool.

If the user asks a question that requires knowledge of coordinates use the 
geocoding tool.
"""

starting_prompt = SystemMessage(content=essential_context)
DS = xr.open_dataset(os.getenv("NETCDF_DATA_PATH"))

# First message
inputs = {"messages": [starting_prompt],
          "dataset": DS
          }

# Initialize token counters
total_input = 0
total_output = 0

# Running graph
for s in app.stream(inputs, stream_mode="values"):
    # Get last message
    message = s["messages"][-1]

    # Display message
    print(format_stream(message))

    # Collect token use information from llm
    if message.type == "ai" and hasattr(message,
                                        "usage_metadata") and message.usage_metadata:
        metadata = message.usage_metadata

        message_input = metadata.get("input_tokens", 0)
        message_output = metadata.get("output_tokens", 0)

        total_input += message_input
        total_output += message_output

# Show the conversation's cumulative token use at the end
token_string = f"|Token consumption: {total_input + total_output}|"
bars = '-' * len(token_string)
print(bars)
print(token_string)
print(bars)
