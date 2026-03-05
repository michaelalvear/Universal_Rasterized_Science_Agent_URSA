"""
Testing tools + state schema integration
"""
# For paths and environment variables
import os
import sys
from dotenv import load_dotenv

# Modules under test
sys.path.append(os.path.abspath("../src"))  # Add "src" dir to the search path
from tools import *
from schemas import *

# Langgraph/Langchain
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

# Xarray
import xarray as xr

# Catch unnecessary "data artifacts" from appearing in terminal
import warnings

warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

# ++++++++++ Initializing Tool Calls ++++++++++
bisect_context_retriever_call = {
    "name": "bisect_context_retriever",
    "args": {
        "query": "Salinity",
    },
    "id": "call_1",
    "type": "tool_call"
}

dataset_metadata_retriever_call = {
    "name": "dataset_metadata_retriever",
    "args": {},
    "id": "call_2",
    "type": "tool_call"
}

spatial_temporal_select_call = {
    "name": "spatial_temporal_select",
    "args": {
        "kwargs": {
            "x": [572125, 592125],
            "y": [2810601, 2830601],
            "time": ["2016-01-01", "2016-01-31"]
        }
    },
    "id": "call_3",
    "type": "tool_call"
}

filter_by_value_call = {
    "name": "filter_by_value",
    "args": {
        "target": "salinity",
        "symbol": ">",
        "value": 35.0
    },
    "id": "call_4",
    "type": "tool_call"
}

resample_time_series_call = {
    "name": "resample_time_series",
    "args": {
        "freq": "1MS",
        "method": "mean"
    },
    "id": "call_5",
    "type": "tool_call"
}

reduce_dimension_call = {
    "name": "reduce_dimension",
    "args": {
        "dim": "time",
        "method": "max"
    },
    "id": "call_6",
    "type": "tool_call"
}

inspect_selection_call = {
    "name": "inspect_selection",
    "args": {},
    "id": "call_7",
    "type": "tool_call"
}

geocoding_tool_call = {
    "name": "geocoding_tool",
    "args": {
        "location_name": "Biscayne National Park"
    },
    "id": "call_8",
    "type": "tool_call"
}

tool_node_1_message = AIMessage(
    content="",
    tool_calls=[bisect_context_retriever_call,
                dataset_metadata_retriever_call,
                spatial_temporal_select_call,
                filter_by_value_call,
                resample_time_series_call,
                reduce_dimension_call,
                geocoding_tool_call]
)

tool_node_2_message = AIMessage(
    content="",
    tool_calls=[inspect_selection_call]
)

# ++++++++++ Graph setup ++++++++++
graph = StateGraph(AgentState)

tool_node_1 = ToolNode(tools=[
    bisect_context_retriever,
    dataset_metadata_retriever,
    spatial_temporal_select,
    filter_by_value,
    resample_time_series,
    reduce_dimension,
    geocoding_tool
])


def new_tool_calls_node(state: AgentState):
    """
    Updates the state with new tool calls for the second phase of tests
    """
    return {"messages": tool_node_2_message}


tool_node_2 = ToolNode(tools=[
    inspect_selection
])

graph.add_node("tool node 1", ursa_tool_node)
graph.add_node("new tool calls node", new_tool_calls_node)
graph.add_node("tool node 2", ursa_tool_node)

graph.add_edge(START, "tool node 1")
graph.add_edge("tool node 1", "new tool calls node")
graph.add_edge("new tool calls node", "tool node 2")
graph.add_edge("tool node 2", END)

app = graph.compile()

# ++++++++++ Check Test Results ++++++++++
load_dotenv()
DS = xr.open_dataset(os.getenv("NETCDF_DATA_PATH"))

initial_state = {
    "messages": [tool_node_1_message],
    "dataset": DS,
}

final_state = app.invoke(initial_state)

print(f"\n{'=' * 30}")
print("TEST START")
print(f"{'=' * 30}\n")
for msg in final_state["messages"]:
    # Skip the initial "AI message", just show the Tool responses
    if msg.type == "tool":
        print(f"\n{'+' * 40}")
        print(f"TOOL: {msg.name.upper() if msg.name else ''}")
        print(f"RESPONDING TO: {msg.tool_call_id}")
        print(f"RESULT:\n{msg.content}")
        print(f"{'+' * 40}\n")

print(f"\n{'=' * 30}")
print("TEST END")
print(f"{'=' * 30}\n")
