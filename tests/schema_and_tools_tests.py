"""Testing tools + state schema integration"""
# For paths and environment variables
import sys
import os
from dotenv import load_dotenv

# Modules under test
sys.path.append(os.path.abspath("../src"))  # Add 'src' dir to the search path
from tools import *
from schemas import *

# Langgraph/Langchain
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

# Xarray
import xarray as xr

# ++++++++++ Initializing state variables ++++++++++
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

see_gallery_contents_call = {
    "name": "see_gallery_contents",
    "args": {},
    "id": "call_5",
    "type": "tool_call"
}

map_slicer_call = {
    "name": "map_slicer",
    "args": {
        "x_min": "582125.24",
        "y_min": "2820601.01",
        "start_time": "2016-01-01"
    },
    "id": "call_3",
    "type": "tool_call"
}

geocoding_tool_call = {
    "name": "geocoding_tool",
    "args": {
        "location_name": "Biscayne National Park"
    },
    "id": "call_4",
    "type": "tool_call"
}

tool_node_message = AIMessage(
    content="",
    tool_calls=[bisect_context_retriever_call,
                dataset_metadata_retriever_call,
                see_gallery_contents_call,
                map_slicer_call,
                geocoding_tool_call]
)

load_dotenv()
DS = xr.open_dataset(os.getenv("NETCDF_DATA_PATH"))

selection = {
    "x": 582125.24,
    "y": 2820601.01,
    "time": "2016-01-01"
}

sample_bundle = SliceBundle(
    map_slice=DS["salinity"].sel(selection, method="nearest"),
    description="sample description",
    tool_origin="sample origin"
)

sample_gallery = MapGallery()

sample_gallery.add_bundle(sample_bundle)

initial_state = {
    "messages": [tool_node_message],
    "dataset": DS,
    "map_gallery": sample_gallery
}

# ++++++++++ Graph setup ++++++++++
graph = StateGraph(AgentState)

tool_node = ToolNode(tools=[map_slicer,
                            geocoding_tool,
                            see_gallery_contents,
                            dataset_metadata_retriever,
                            bisect_context_retriever])


def gallery_update_node(state: AgentState):
    '''
    Finds all ToolMessages in the most recent update that contain
    a SliceBundle artifact and pushes them to the gallery.
    '''

    # Get the current message list
    messages = state.messages
    if not messages:
        return {}

    # Look at the last ToolMessages to look for "artifact" fields

    new_bundles = []
    # Iterate backwards through the messages until we hit the AI's request
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            # Check if this specific tool message has a bundle
            if isinstance(msg.artifact, SliceBundle):
                new_bundles.append(msg.artifact)
        else:
            # We reached the AIMessage that triggered these tools
            break

    # Return the update
    if new_bundles:
        update_gallery = MapGallery()
        for bundle in reversed(new_bundles):
            update_gallery.add_bundle(bundle)
        # We reverse them back so they are in chronological order
        return {"map_gallery": update_gallery}
    else:
        return {}


graph.add_node("tool node", tool_node)
graph.add_node("gallery update node", gallery_update_node)

graph.add_edge(START, "tool node")
graph.add_edge("tool node", "gallery update node")
graph.add_edge("gallery update node", END)

app = graph.compile()

# ++++++++++ Check test results ++++++++++
print(f"\n{'=' * 30}")
print("TEST START")
print(f"{'=' * 30}\n")

final_state = app.invoke(initial_state)
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

# ++++++++++ See final gallery contents ++++++++++
print(f"\n{'=' * 30}")
print("FINAL GALLERY:")
print(f"{'=' * 30}\n")

for index, bundle in final_state["map_gallery"].bundles.items():
    print(f"\n{'+' * 40}")
    print(f"INDEX: {index}")
    print(f"DESCRIPTION: {bundle.description}")
    print(f"TOOL ORIGIN: {bundle.tool_origin}")
    print(f"{'+' * 40}\n")
