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

# Add utilities to search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "../utilities")))
from message_formatter import format_msg

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

def user_input(state: AgentState) -> dict[str, List[HumanMessage]]:
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
llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0,
                             streaming=True).bind_tools(tools)


# Main llm invocation
def llm_call(state: AgentState) -> dict[str, List[AIMessage]]:
    """
    LLM decides whether to call a tool or not.
    """

    llm_response = llm.invoke(state.messages)
    return {"messages": [llm_response]}


# llm to tool node routing function
def tool_router(state: AgentState) -> Literal["pending tool calls", "done"]:
    """
    Decide if we should continue the tool loop or return to the user based on
    the last message.
    """
    last_message = state.messages[-1]

    # Continue tool loop
    if last_message.tool_calls:
        return "pending tool calls"

    # No tools are called -> return to user node
    return "done"


graph = StateGraph(AgentState)

# Debug mode graph
if __name__ == "__main__":

    # Nodes
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

# Flask server graph
else:

    # Nodes
    graph.add_node("llm call", llm_call)
    graph.add_node("tool node", ursa_tool_node)

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
*ALWAYS RETURN A NATURAL LANGUAGE MESSAGE WHEN 
INVOKED, EVEN WHEN MAKING TOOL CALLS, EXPLAIN YOUR THOUGHT PROCESS* 

*REMEMBER: DON'T END RESPONSES WITH NEWLINES, TABS, OR SPACES*

You are a helpful assistant tasked with retrieving and interpreting information 
from a South Florida hydrological model known as:
Biscayne and Southern Everglades Coastal Transport Model (BISECT).
 
Details about the model are recorded in the paper:
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
provided too you. 
Follow argument schemas *EXACTLY*. 

*DON'T MAKE ASSUMPTIONS ABOUT THE STRUCTURE OF THE DATASET*
*ALWAYS MANUALLY CHECK METADATA*

You are a data interface. You are *FORBIDDEN* from providing numerical data (
salinity, means, ranges) unless you have successfully received a ToolMessage 
containing that specific data in the current conversation turn. 

Nan values indicate a near zero salinity measurement typically indicating the
presence of a landmass.

Units have been converted from the original PSU units of the model into grams 
per liter. 

*REMEMBER: THE UNITS ARE GRAMS PER LITER*

The updated data after each operation is preserved in your state, so if you 
need to perform a multistep operation you can.

To see the a statistical/geospatial summary of the extracted data held in 
your active selection use the inspect_selection tool.

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

# ++++++++++ Console Debug Mode ++++++++++
if __name__ == "__main__":

    # Initialize state
    inputs = {"messages": [SystemMessage(content=essential_context)],
              "dataset": DS
              }

    # Initialize token counter
    total_tokens = 0

    # Streaming agent
    # ('updates' mode yields the state updates after each node execution)
    for update in app.stream(inputs, stream_mode="updates"):

        for node_name, state_update in update.items():
            if "messages" in state_update:
                new_msgs = state_update["messages"]

                for msg in new_msgs:
                    print(format_msg(msg))

                    # Count tokens
                    if isinstance(msg, AIMessage) and getattr(msg,
                                                              "usage_metadata",
                                                              None):
                        total_tokens += msg.usage_metadata.get("total_tokens",
                                                               0)

    # Show the conversation's cumulative token use at the end
    token_string = f"|Token consumption: {total_tokens}|"
    bars = '-' * len(token_string)
    print(bars)
    print(token_string)
    print(bars)

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
