"""These are the tools provided to the Agent"""

# Environment variable access
import os
from dotenv import load_dotenv

# Essential LangChain/LangGraph packages
from langchain_core.tools import tool
from langchain_core.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain.tools import InjectedState

# Chroma/RAG
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Geocoding
from geopy.geocoders import GoogleV3
from pyproj import Transformer

# Type hinting/validation
from pydantic import BaseModel, Field
from typing import Annotated, Optional, Dict, List, Tuple, Any
from xarray import Dataset
from datetime import date as dt_date

# Schemas
from schemas import SliceBundle, MapGallery

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
bisect_context_retriever = create_retriever_tool(retriever=retriever,
                                                 name="bisect_context_retriever",
                                                 description="Search and return relevant portions of the bisect paper.",
                                                 document_prompt=custom_doc_prompt,
                                                 response_format="content")


# ++++++++++++++++++++ Dataset Metadata Retrieval ++++++++++++++++++++

@tool("dataset_metadata_retriever")
def dataset_metadata_retriever(
        # Gets the entire src Dataset
        ds: Annotated[Dataset, InjectedState("dataset")]
) -> str:
    """This tool allows you to see the metadata of the entire dataset"""
    metadata = str(ds)
    return metadata


# ++++++++++++++++++++ Gallery Context ++++++++++++++++++++

# See gallery contents
@tool("see_gallery_contents")
def see_gallery_contents(
        # Gets the current map gallery from the state
        gallery: Annotated[MapGallery, InjectedState("map_gallery")]
) -> Dict[str, List[str]]:
    """
    Returns a dictionary describing
    the indices and contents in the map gallery
    """

    gallery_description = {}

    """This gives a description of what's currently in the map gallery"""
    for index, bundle in gallery.bundles.items():
        key = f"INDEX {index}: "
        value = [f"DESCRIPTION: {bundle.description}",
                 f"TOOL_ORIGIN: {bundle.tool_origin}"]
        gallery_description[key] = value

    return gallery_description


# ++++++++++++++++++++ Populate Map Gallery ++++++++++++++++++++
class MapSlicerInput(BaseModel):
    """Input schema for map slicer"""
    x_min: float = Field(description="X-coordinate start.")
    x_max: Optional[float] = Field(default=None,
                                   description="X-coordinate end (only for ranges).")

    y_min: float = Field(description="Y-coordinate start.")
    y_max: Optional[float] = Field(default=None,
                                   description="Y-coordinate end (only for ranges).")

    start_time: dt_date = Field(description="Start date (YYYY-MM-DD).")
    end_time: Optional[dt_date] = Field(default=None,
                                        description="End date (only for ranges).")


@tool("map_slicer", args_schema=MapSlicerInput,
      response_format="content_and_artifact")
def map_slicer(
        # Automagically gets the corresponding tool call ID
        tool_call_id: Annotated[str, InjectedToolCallId],
        # Fetches the whole xarray dataset from the state
        ds: Annotated[Dataset, InjectedState("dataset")],
        x_min: float,
        y_min: float,
        start_time: dt_date,
        x_max: Optional[float] = None,
        y_max: Optional[float] = None,
        end_time: Optional[dt_date] = None,
) -> Tuple[str, Any]:
    """
    Use this tool to add a map slice to your map gallery.
    - Provide only 'min' values for a single point.
    - Provide 'min' and 'max' values for a range or area.
  """

    # Helper to determine if we are slicing or selecting
    def get_selector(min, max):
        if max == None:
            return min
        else:
            return slice(min, max)

    try:
        # Dynamically build the selection dictionary
        selection = {
            "x": get_selector(x_min, x_max),
            "y": get_selector(y_min, y_max),
            "time": get_selector(start_time, end_time)
        }

        # Apply selection
        map_slice = ds["salinity"].sel(selection, method="nearest")

        # Create a description for the new map slice
        description = "Metadata:\n" + str(map_slice)

        new_slice_bundle = SliceBundle(map_slice=map_slice,
                                       description=description,
                                       tool_origin="map_slicer")

        content = "Slice created succesfully"
        artifact = new_slice_bundle
        return content, artifact

    except Exception as e:
        content = f"Query failed: {str(e)}"
        artifact = None

        return content, artifact


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
UTM_17_EASTING_RANGE = (
    160000, 840000)  # Constants for UTM Zone 17N valid range


@tool("geocoding_tool", args_schema=GeocodingInput)
def geocoding_tool(location_name: str) -> dict[str, Any]:
    """
    Useful for finding UTM Zone 17N coordinates
    when you only have a place name.
    """

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    geolocator = GoogleV3(api_key=api_key)

    location = geolocator.geocode(location_name)

    if not location:
        return {"error": f"Could not find {location_name}"}

    easting, northing = latlon_to_utm17(location.latitude, location.longitude)

    # If easting is in the proper UTM 17 range (A.K.A around Florida longitude)
    if UTM_17_EASTING_RANGE[0] <= easting <= UTM_17_EASTING_RANGE[1]:
        return {
            "easting": round(easting, 2),
            "northing": round(northing, 2),
            "found_address": location.address
        }
    else:
        return {
            "error": f"Location '{location.address}' is outside the valid UTM 17N zone.",
            "easting": easting,
            "northing": northing
        }
