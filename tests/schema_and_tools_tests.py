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
            "time": ["2096-01-01", "2096-01-31"]
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

inspect_selection_call_1 = {
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

reset_view_call = {
    "name": "reset_view",
    "args": {},
    "id": "call_9",
    "type": "tool_call"
}

inspect_selection_call_2 = {
    "name": "inspect_selection",
    "args": {},
    "id": "call_10",
    "type": "tool_call"
}

tool_node_message = AIMessage(
    content="",
    tool_calls=[bisect_context_retriever_call,
                dataset_metadata_retriever_call,
                spatial_temporal_select_call,
                filter_by_value_call,
                resample_time_series_call,
                reduce_dimension_call,
                inspect_selection_call_1,
                geocoding_tool_call,
                reset_view_call,
                inspect_selection_call_2
                ]
)


# ++++++++++ Graph setup ++++++++++
graph = StateGraph(AgentState)

graph.add_node("tool node", ursa_tool_node)

graph.add_edge(START, "tool node")
graph.add_edge("tool node", END)

app = graph.compile()
# ++++++++++ Check Test Results ++++++++++
load_dotenv()
DS = xr.open_dataset(os.getenv("NETCDF_DATA_PATH"))
my_tools = generate_tools(DS)

initial_state = {
    "messages": [tool_node_message],
    "dataset": DS,
    "tools": my_tools
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
