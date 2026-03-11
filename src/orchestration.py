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
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../utilities")))
from stream_formatter import format_stream

# Langgraph/Langchain
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Xarray
import xarray as xr
import numpy as np

# Types
from typing import Literal, List

load_dotenv()  # Load environment variables

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


tools = [
    bisect_context_retriever,
    dataset_metadata_retriever,
    spatial_temporal_select,
    filter_by_value,
    resample_time_series,
    reduce_dimension,
    inspect_selection,
    reset_view,
    geocoding_tool
]

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

graph.add_node("tool node", ursa_tool_node)
graph.add_node("llm call", llm_call)

# Edges
graph.add_edge(START, "llm call")

graph.add_conditional_edges(
    "llm call",
    tool_router,
    {
        "pending tool calls": "tool node",
        "done": END
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

Your can get context on the paper through the tools provided to you.

Reflect on any context you fetch, and keep retrieving until you have sufficient 
context to answer the user's research request.

*ALWAYS PROVIDE COMPLETE CITATIONS*

You can also extract a subset of the raster data using the GIS tool suite 
provided too you for your own reference during the course of conversation. 
Follow argument schemas *EXACTLY*. 

The updated data after each operation is preserved in your state, so if you 
need to perform a multistep operation you can.

To see the a statistical summary of the extracted data held in your active 
selection use the inspect_selection tool.

To reset your view of the data back to the original data so you can run new
operations use the reset_view tool.

If the user asks a question that requires knowledge of coordinates use the
geocoding tool.

Use plain text only in your responses. Do not use markdown formatting such as
**bold**, *italic*, or # headers.

TOOL USE RULES — follow these strictly:
- If the user asks to "show", "plot", "map", "chart", "graph", or "visualize"
  any data, you MUST use the GIS tools to extract the data. Do not just
  describe it in text.
- For a time series at a location (e.g. "show salinity at X over time"):
  use geocoding_tool to get coordinates, then spatial_temporal_select to
  pin x and y to that point and select the time range.
- For a spatial map (e.g. "show a map of salinity"):
  use reduce_dimension with dim="time" to collapse time first.
- Always call inspect_selection after extracting data to confirm the result
  before giving your final response.
"""

DS = xr.open_dataset(os.getenv("NETCDF_DATA_PATH"))

# ++++++++++ Flask Interface ++++++++++

def run_agent(user_message: str, history: list = None) -> dict:
    """
    Takes a user's natural language message, runs it through the agent,
    and returns a dict: {text, chart_type, labels, data}.
    Called by app.py's /query route.

    history: list of {"role": "user"|"assistant", "content": str} dicts
             representing prior turns in the conversation.
    """
    # Flatten history into a text block prepended to the current message.
    # Injecting bare AIMessage objects into LangGraph's initial state causes
    # the add_messages reducer to fail as history grows.
    if history:
        history_text = "\n".join(
            f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}"
            for t in history
        )
        full_message = f"[Previous conversation]\n{history_text}\n\n[Current message]\n{user_message}"
    else:
        full_message = user_message

    initial_state = {
        "messages": [
            SystemMessage(content=essential_context),
            HumanMessage(content=full_message)
        ],
        "dataset": DS
    }

    result = app.invoke(initial_state)

    # Extract text, handling Gemini's list-of-blocks format
    content = result["messages"][-1].content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    # Extract chart data from active_selection.
    # active_selection always has a value (defaults to full DS via model_validator).
    # We detect a meaningful slice by checking remaining dims:
    #   - Full dataset has 3 dims: ["time", "y", "x"] → no chart
    #   - 1D ["time"] → LLM pinned to a location, kept time range → line chart
    #   - 2D ["y", "x"] → LLM collapsed time → heatmap
    chart_type, labels, data = None, [], []
    # Use .get() because active_selection only appears in result if a GIS tool
    # ran and explicitly updated it. Text-only responses leave it absent.
    # Fallback to DS gives 3 dims → no chart, which is correct behavior.
    selection = result.get("active_selection", DS)
    dims = set(selection.dims)

    if dims == {"time"}:
        # Time series at a single point
        chart_type = "line"
        labels = [str(t)[:10] for t in selection["time"].values]
        raw = selection["salinity"].values.flatten()
        data = [None if np.isnan(v) else round(float(v), 4) for v in raw]

    elif dims == {"x", "y"}:
        # Spatial map, time collapsed
        total = selection.sizes["x"] * selection.sizes["y"]
        if total <= 1000000000:
            chart_type = "heatmap"
            sal = selection["salinity"].transpose("y", "x")
            x_coords = [round(float(v), 0) for v in selection["x"].values]
            y_coords = [round(float(v), 0) for v in selection["y"].values]
            grid = sal.values
            flat = []
            for yi in range(len(y_coords)):
                for xi in range(len(x_coords)):
                    v = grid[yi, xi]
                    flat.append({
                        "x": xi,
                        "y": yi,
                        "v": None if np.isnan(v) else round(float(v), 4)
                    })
            labels = {"x": x_coords, "y": y_coords}
            data = flat

    return {
        "text": content,
        "chart_type": chart_type,
        "labels": labels,
        "data": data
    }


# ++++++++++ Console Runner ++++++++++

if __name__ == "__main__":
    starting_prompt = SystemMessage(content=essential_context)

    inputs = {"messages": [starting_prompt],
              "dataset": DS
              }

    total_tokens = 0

    for s in app.stream(inputs, stream_mode="values"):
        message = s["messages"][-1]
        print(format_stream(message))

        if message.type == "ai" and hasattr(message,
                                            "usage_metadata") and message.usage_metadata:
            metadata = message.usage_metadata
            total_tokens += metadata.get("total_tokens", 0)

    token_string = f"|Token consumption: {total_tokens}|"
    bars = '-' * len(token_string)
    print(bars)
    print(token_string)
    print(bars)
