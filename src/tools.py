"""These are the tools provided to the Agent"""

# Environment variable access
import os
from dotenv import load_dotenv

# Essential LangChain/LangGraph packages
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

# Chroma/RAG
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Geocoding
from geopy.geocoders import Nominatim
from pyproj import Transformer

# Type hinting/validation
from typing import Literal, Union, Dict, Any
from langchain_core.messages import AIMessage, ToolMessage
import xarray as xr
from inspect import signature
from schemas import *

# Operators
import operator

load_dotenv()  # Load environment variables

# ++++++++++++++++++++ RAG Retrieval ++++++++++++++++++++
# Setting up vector store access
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Connect to the existing DB directory
vectorstore = Chroma(
    persist_directory=os.getenv("CHROMADB_PATH"),
    embedding_function=embeddings,
    collection_name="BISECT"
)

# Creating the retriever, retrieves records in the form of "Document" objects
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # K is the amount of docs to return
)

# This template is passed to create_retriever_tool.
# This tells the function how to stringify the text
# of the retrieved Document objects along with their metadata (page_label)
custom_doc_prompt = PromptTemplate.from_template(
    "--- DOCUMENT CHUNK ---\n"
    "SOURCE PAGE: {page_label}\n"
    "CONTENT: {page_content}\n"
)

# This is the actual StructuredTool
# A function that formats retriever output into a string
bisect_context_retriever = create_retriever_tool(
    retriever=retriever,
    name="bisect_context_retriever",
    description="Search and return relevant portions of the bisect paper.",
    document_prompt=custom_doc_prompt,
    response_format="content"
)


# ++++++++++++++++++++ Dataset Metadata Retrieval ++++++++++++++++++++

@tool("dataset_metadata_retriever")
def dataset_metadata_retriever(
        # Gets the entire src Dataset
        dataset: Annotated[xr.Dataset, InjectedState]
) -> str:
    """This tool allows you to see the metadata of the entire dataset"""
    ds = dataset
    metadata = str(ds)
    return metadata


# ++++++++++++++++++++ See Selection Details ++++++++++++++++++++
@tool("inspect_selection")
def inspect_selection(
        current_ds: Annotated[Optional[xr.Dataset], InjectedState]
) -> str:
    """
    Statistical summary of the active_slice.
    Global filters are assumed to handle DOF/NaN warnings.
    """
    ds = current_ds
    if ds is None:
        return "No active selection found."

    summary = {}

    for var in ds.data_vars:
        da = ds[var]

        # If the slice is empty (0 pixels), don't even try math
        if da.size == 0:
            summary[var] = {"error": "Empty selection"}
            continue

        # Simple, direct calculations
        summary[var] = {
            "mean": round(float(da.mean()), 2),
            "max": round(float(da.max()), 2),
            "min": round(float(da.min()), 2),
            "std": round(float(da.std()), 2),
            "units": da.attrs.get("units", "unknown"),
            "long_name": da.attrs.get("long_name", var),
            "null_percentage": round(float(da.isnull().mean() * 100), 1)
        }

    coords_info = {
        dim: {
            "size": ds.sizes[dim],
            "range": [float(ds[dim].min()), float(ds[dim].max())]
        } for dim in ds.dims
    }

    return f"Data Profile: {{'variable_stats': {summary}, 'coordinates': {coords_info}}}"


# ++++++++++++++++++++ Changing View ++++++++++++++++++++

# Define the specific "Vocabulary" of your NetCDF file This prevents the
# LLM from hallucinating variable names like 'temp' or 'depth'

Variable = Literal["salinity"]
Dimension = Literal["x", "y", "time"]
MathSymbol = Literal[">", "<", ">=", "<=", "==", "!="]
StatsMethod = Literal["mean", "max", "min", "std", "median"]
TimeFreq = Literal[
    "1D", "1W", "1MS", "1YS"]  # Daily, Weekly, Month Start, Year Start


class SpatialTemporalSelectSchema(BaseModel):
    kwargs: Dict[
        Dimension, Union[float, str, List[Union[float, str]]]] = Field(
        ...,
        description="Coordinate constraints. Use [min, max] for a range or a "
                    "single value. Takes spatial (x, y) and calendar dates "
                    "for time.",
        example={"x": [580000, 585000], "time": "2026-01-01"}
    )


class FilterByValueSchema(BaseModel):
    target: Variable = Field(...,
                             description="The variable to filter (e.g., "
                                         "'salinity').")
    symbol: MathSymbol = Field(..., description="The comparison operator.")
    value: float = Field(..., description="The threshold value.")


class ResampleTimeSeriesSchema(BaseModel):
    freq: TimeFreq = Field(..., description="Temporal grouping frequency.")
    method: StatsMethod = Field(...,
                                description="Aggregation method (e.g., "
                                            "'mean').")


class ReduceDimensionSchema(BaseModel):
    dim: Dimension = Field(..., description="The dimension to collapse.")
    method: StatsMethod = Field(..., description="The reduction method.")


@tool(name_or_callable="spatial_temporal_select",
      args_schema=SpatialTemporalSelectSchema)
def spatial_temporal_select(
        kwargs: Dict[Dimension, Union[float, str, List[Union[float, str]]]],
        current_ds: Annotated[xr.Dataset, InjectedState]
) -> xr.Dataset:
    """Slices the dataset by space (x, y) or time coordinates.
    Use for subsetting."""
    ds = current_ds
    slices = {}
    points = {}

    for k, v in kwargs.items():
        if isinstance(v, list):
            # Check coordinate direction for xarray slice
            if ds[k].values[0] > ds[k].values[-1]:
                slices[k] = slice(max(v), min(v))
            else:
                slices[k] = slice(min(v), max(v))
        else:
            points[k] = v

    if slices:
        ds = ds.sel(slices)
    if points:
        ds = ds.sel(points, method="nearest")

    return ds


@tool(name_or_callable="filter_by_value", args_schema=FilterByValueSchema)
def filter_by_value(
        target: Variable,
        symbol: MathSymbol,
        value: float,
        current_ds: Annotated[xr.Dataset, InjectedState]
) -> xr.Dataset:
    """Applies a mask to data based on values (e.g., keep salinity > 30)."""
    ds = current_ds
    # This dictionary maps string symbols to functional logic.
    ops = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne
    }

    condition = ops[symbol](ds[target], value)
    new_ds = ds.copy()
    new_ds[target] = new_ds[target].where(condition)
    return new_ds


@tool(name_or_callable="resample_time_series",
      args_schema=ResampleTimeSeriesSchema)
def resample_time_series(
        freq: TimeFreq,
        method: StatsMethod,
        current_ds: Annotated[xr.Dataset, InjectedState]
) -> xr.Dataset:
    """Aggregates the time dimension into larger bins (e.g., Monthly Mean)."""
    ds = current_ds
    resampler = ds.resample(time=freq)
    return getattr(resampler, method)()


@tool(name_or_callable="reduce_dimension", args_schema=ReduceDimensionSchema)
def reduce_dimension(
        dim: Dimension,
        method: StatsMethod,
        current_ds: Annotated[xr.Dataset, InjectedState]
) -> xr.Dataset:
    """Collapses a dimension entirely. Use 'time' to create a map,
    or 'x'/'y' for profiles."""
    ds = current_ds
    return getattr(ds, method)(dim=dim)


@tool(name_or_callable="reset_view")
def reset_view(dataset: Annotated[xr.Dataset, InjectedState]):
    """Resets the active view of the data back to the original dataset,
    so you can make new queries."""
    ds = dataset
    return ds


# ++++++++++++++++++++ Geocoding ++++++++++++++++++++

# Pyproj transformer helper function
# This transformer can only make sense of values that
# lie within UTM Zone 17N (Florida area)
_latlon_to_utm17 = Transformer.from_crs(
    "EPSG:4326",  # WGS84 lat/lon
    "EPSG:26917",  # NAD83 / UTM Zone 17N
    always_xy=True
)


# Wrapping transformer in a function
def latlon_to_utm17(lat: float, lon: float) -> tuple[Any, Any]:
    """
    Convert latitude and longitude to UTM meters (EPSG:26917).

    The return values follows the xy convention: easting followed by northing

    Parameters
    ----------
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees

    Returns
    -------
    x : float
        UTM Easting (meters)
    y : float
        UTM Northing (meters)
    """

    x, y = _latlon_to_utm17.transform(lon, lat, errcheck=True)
    return x, y


# Input schema
class GeocodingInput(BaseModel):
    """"Input schema for geocoding tool"""
    location_name: str = Field(
        ...,
        description="The name of the location to look up (e.g., 'Biscayne Bay')"
    )


# Tool logic

# --- BISECT model bounds ---
left, right = 461000.0, 590500.0
top, bottom = 2872000.0, 2779000.0


@tool("geocoding_tool", args_schema=GeocodingInput)
def geocoding_tool(location_name: str) -> dict[str, Any]:
    """
    Useful for finding UTM Zone 17N coordinates
    when you only have a place name.
    """

    geolocator = Nominatim(user_agent="ursa_hydrology")
    location = geolocator.geocode(location_name)

    if not location:
        return {"error": f"Could not find {location_name}"}

    easting, northing = latlon_to_utm17(location.latitude, location.longitude)

    # Check If coordinates are in the proper range
    # (A.K.A around South Florida )
    if left <= easting <= right and bottom <= northing <= top:
        return {
            "easting": round(easting, 2),
            "northing": round(northing, 2),
            "found_address": location.address
        }
    else:
        return {
            "alert": f"Location '{location.address}' is outside the valid "
                     f"UTM 17N zone. Coordinate projection may be inaccurate.",
            "easting": easting,
            "northing": northing
        }


# ++++++++++++++++++++ Custom Tool Node  ++++++++++++++++++++
def ursa_tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes tools that return xr.Datasets and updates the graph state.
    """
    tools_by_name = {
        t.name: t for t in [
            bisect_context_retriever,
            dataset_metadata_retriever,
            spatial_temporal_select,
            filter_by_value,
            resample_time_series,
            reduce_dimension,
            reset_view,
            inspect_selection,
            geocoding_tool
        ]
    }

    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}

    new_messages = []
    current_ds = state.active_selection
    argument_map = {
        "dataset": state.dataset,
    }

    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        call_id = tool_call["id"]

        if name not in tools_by_name:
            new_messages.append(ToolMessage(content=f"Error: {name} not found",
                                            tool_call_id=call_id))
            continue

        argument_map["current_ds"] = current_ds

        current_tool = tools_by_name[name]

        # Start with arguments provided by the LLM
        final_args = tool_call["args"].copy()

        # Get complete list of tool parameters including injected params
        sig = signature(current_tool.func)
        expected_params = sig.parameters.keys()

        # Loop through our mapping and fill missing pieces
        for param_name, data_value in argument_map.items():
            if param_name in expected_params:
                # We inject the data only if the tool is asking for it by name
                final_args[param_name] = data_value

        try:
            # Execute tool
            # We use .invoke and manually inject our args
            result = current_tool.invoke(final_args)

            # Update the tracking variable so the next tool call in the loop
            # sees the result of this one.
            if isinstance(result, (xr.Dataset, xr.DataArray)):

                current_ds = result

                # Make sure the result of a tool is a DataSet in case it gets
                # flattened to a DataArray in one of the tools
                current_ds = current_ds.to_dataset() if isinstance(current_ds,
                                                                   xr.DataArray) else current_ds

                # Provide a high-level summary as the tool's response to the
                # LLM
                summary = f"Operation successful. New shape: {dict(current_ds.sizes)}"
                new_messages.append(
                    ToolMessage(content=summary, tool_call_id=call_id,
                                name=name))

            else:
                new_messages.append(
                    ToolMessage(content=str(result), tool_call_id=call_id,
                                name=name))

        except Exception as e:
            new_messages.append(
                ToolMessage(content=f"Error: {str(e)}", tool_call_id=call_id,
                            name=name))

    # Return the final modified dataset and the messages
    return {
        "active_selection": current_ds, # This overwrites the state with the final result
        "messages": new_messages  # This appends the tool feedback to history
    }
