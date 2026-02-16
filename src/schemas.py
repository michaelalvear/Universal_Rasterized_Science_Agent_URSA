"""
This file holds pydantic models related to the Langgraph Agent's State
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated, List, Dict
from xarray import Dataset, DataArray
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SliceBundle(BaseModel):
    """This holds a map slice and it's metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    map_slice: DataArray
    description: str
    tool_origin: str


class MapGallery(BaseModel):
    """This datastructure is an indexed dictionary of map slices."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bundles: Dict[int, SliceBundle] = Field(default_factory=dict)
    next_id: int = 0

    def add_bundle(self, new_bundle: SliceBundle) -> int:
        """Adds a new map slice bundle and returns the assigned ID."""
        assigned_id = self.next_id
        self.bundles[assigned_id] = new_bundle
        self.next_id += 1
        return assigned_id


def merge_gallery(current: MapGallery, update: MapGallery) -> MapGallery:
    """
    Takes the current MapGallery and adds the bundles from a new one,
    uses the internal add_slice method to update it, and returns the new state.
    """

    # Creates a copy of the current map gallery
    new_gallery = current.model_copy(deep=True)

    # Adds new slices
    for bundle in update.bundles.values():
        new_gallery.add_bundle(bundle)

    # A maximum of 5 slices allowed at a time to preserve memory
    if len(new_gallery.bundles) > 5:
        # Keep only the 5 highest IDs
        sorted_ids = sorted(new_gallery.bundles.keys())
        new_gallery.bundles = {k: new_gallery.bundles[k] for k in
                               sorted_ids[-5:]}

    return new_gallery


class AgentState(BaseModel):
    """State schema"""
    # This holds the conversation history
    messages: Annotated[List[BaseMessage], add_messages]

    # Enable arbitrary types so Pydantic accepts Xarray types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # This holds the main dataset
    dataset: Dataset

    # This holds slices of the map necessary to answer the users question
    map_gallery: Annotated[MapGallery, merge_gallery]