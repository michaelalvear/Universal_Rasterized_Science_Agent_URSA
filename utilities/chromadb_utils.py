"""
This file allows you to read, write, and delete documents to a chromadb
vector database.

embedding model = gemini-embedding-001

Author: Michael Alvear
Date: 1/12/2026

Partially plagiarised from: https://github.com/iamvaibhavmehra/LangGraph
-Course-freeCodeCamp/blob/main/Agents/RAG_Agent.py
"""

# Environment variable access
import os
from dotenv import load_dotenv

# Main Chroma package
import chromadb
# Gemini embedding function
from chromadb.utils.embedding_functions import GoogleGenaiEmbeddingFunction

# For type hinting function parameters
from chromadb.api import ClientAPI
from chromadb.api.types import EmbeddingFunction
from typing import List

# PDF Utilities from LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import uuid  # Used to give each record in vector DB a unique ID


# Embed pdf helper function
def add_pdf(pdf_path: str, collection_name: str, chunk_size: int,
            chunk_overlap: int, chroma_client: ClientAPI,
            embedding_function: EmbeddingFunction[List[str]]) -> None:
    """Adds a PDF to a collection in a Chroma vector database"""

    # Validate path
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Load PDF
    try:
        # Initialize the loader
        pdf_loader = PyPDFLoader(
            file_path=pdf_path)  # Load by page preserves src page metadata

        pages = pdf_loader.load()  # Generates a list of type 'Document'

        print(f"PDF has been loaded and has {len(pages)} pages")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        raise

    # Split text into chunks
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        # The LangChain text splitter preserves src page for each chunk
        pages_split = text_splitter.split_documents(
            pages)
    except Exception as e:
        print(f"Text splitter error: {e}")
        raise

    # Creating unique IDs
    ids = [str(uuid.uuid4()) for _ in range(len(pages_split))]

    # Extracting the contents and metadata for Chroma (from Document objects)
    documents = [doc.page_content for doc in pages_split]
    metadatas = [doc.metadata for doc in pages_split]

    # Add the new vectors to the database (in batches which is safer)
    try:

        # Getting/creating collection
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

        batch_size = 90

        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i: i + batch_size],
                documents=documents[i: i + batch_size],
                metadatas=metadatas[i: i + batch_size]
            )

    except Exception as e:
        print(f"Chroma exception: {e}")

    print("PDF embedded successfully!")


# Load environment variables
load_dotenv()
persist_directory = os.getenv("CHROMADB_PATH")
pdf_path = os.getenv("PDF_PATH")

# Initialize embedding function (this expects your API key in GOOGLE_API_KEY)
embedding_model = "gemini-embedding-001"
google_ef = GoogleGenaiEmbeddingFunction(
    model_name=embedding_model
)

# Initializing ChromaDB client
client = chromadb.PersistentClient(path=persist_directory)

# Get operation
operation = input(
    "say \"add\" to embed | \"delete\" to delete collections | \"preview\" to see documents: "
)

# Execute operation
collections_list = client.list_collections()
if operation == "add":

    add_pdf(pdf_path=pdf_path, collection_name="BISECT", chunk_size=1000,
            chunk_overlap=200, chroma_client=client,
            embedding_function=google_ef)

elif operation == "delete":

    for collection in collections_list:
        client.delete_collection(name=collection.name)

    print("All collections deleted")

elif operation == "preview" and collections_list:

    for collection in collections_list:
        print(f"____________BEGINNING OF {collection.name} SAMPLES____________\n")

        samples = collection.get(limit=5)

        formatted_records = list(zip(samples["ids"], samples["documents"]))

        for id, doc in formatted_records:
            print(f"ID:\n {id}\n")
            print(f"DOCUMENT:\n {doc}")
            print("========================")

        print(f"____________END OF {collection.name} SAMPLES____________")

else:
    print("nothing happened")
