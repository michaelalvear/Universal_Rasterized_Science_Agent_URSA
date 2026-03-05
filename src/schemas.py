"""
This file holds pydantic models related to the Langgraph Agent's State
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated, List, Dict, Optional, Any
from xarray import Dataset, DataArray
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(BaseModel):
    """State schema for the Hydrology Agent"""

    # Conversation History
    messages: Annotated[List[BaseMessage], add_messages]

    # Allow Xarray types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Ensure this is opened lazily (e.g., xr.open_dataset(path, chunks={}))
    dataset: Dataset = Field(
        description="The primary NetCDF lazy-loaded dataset")

    # This gets overwritten every time the agent decides to change the view
    active_selection: Optional[Dataset] = Field(
        None,
        description="The current focused data slice being analyzed or "
                    "visualized"
    )