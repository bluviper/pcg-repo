"""
title: Llama Index RAG Pipeline # Give it a clear title for the UI
author: Your Name # Use your name
date: 2025-07-31 # Updated date for clarity
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index
"""

from typing import List, Union, Generator, Iterator

# LlamaIndex imports
import sys
import os

# --- ADD THESE LINES FOR DEBUGGING sys.path ---
# Try adding common paths where 'schemas' might be found within the container
sys.path.insert(0, '/app/backend/app') # Common for open-webui backend structure
sys.path.insert(0, '/usr/local/lib/python3.11/site-packages/openwebui_backend/app') # Possible installed path
sys.path.insert(0, '/app') # Fallback if schemas is directly under /app
# --- END DEBUGGING sys.path ---

from schemas import OpenAIChatMessage # <-- Keep this UNCOMMENTED. It should work now.
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# If you plan to use Qdrant directly as the vector store for LlamaIndex, uncomment these:
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient

class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None

    async def on_startup(self):

        # Define the persistent storage path inside the container
        PERSIST_DIR = "./index_storage" # Maps to your host's ./index_storage

        # Ensure this API key is consistent (from docker-compose env)
        #os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "dummy_key_if_not_set")
        os.environ["OPENAI_API_KEY"] = "" # Explicitly set to empty string

        try:
            # Configure LlamaIndex to use Ollama
            # IMPORTANT: Ensure 'mistral:7b-instruct-v0.2-q4_K_M' is the exact model name pulled in Ollama
            Settings.llm = Ollama(model_name="mistral:7b-instruct-v0.2-q4_K_M", base_url="http://portable-ollama:11434")
            
            Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://portable-ollama:11434")
            
             # Check current sys.path for debugging
            for p in sys.path:
                print(f"  - {p}")

            # --- Logic for persistent index storage ---
            # Try to load index from storage, otherwise build and save
            if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR): # Check if dir exists AND is not empty
                
                # Load documents. Make sure your /docs folder is populated on the host.
                self.documents = SimpleDirectoryReader("./docs").load_data()

                # If using LlamaIndex's default filesystem persistence (recommended to start):
                self.index = VectorStoreIndex.from_documents(self.documents)
                
                # Save the index to the mounted persistent directory
                self.index.storage_context.persist(persist_dir=PERSIST_DIR)

                # If you want to use Qdrant as the *primary* vector store for persistence,
                # you would need to uncomment the Qdrant related imports and logic:
                # client = QdrantClient(host="portable-qdrant", port=6335)
                # vector_store = QdrantVectorStore(client=client, collection_name="your_rag_collection")
                # storage_context = StorageContext.from_defaults(vector_store=vector_store)
                # self.index = VectorStoreIndex.from_documents(self.documents, storage_context=storage_context)
                # print("--- LlamaIndex vector store index created with Qdrant. ---")
                # # No need to call .persist() explicitly for Qdrant, as it's already persistent.
            else:
                # Load the index from the mounted persistent directory
                self.index = load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))

        except Exception as e:
            import traceback
            traceback.print_exc() # Print full traceback
            raise # Re-raise to ensure the server process doesn't just hang silently

        pass

    async def on_shutdown(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print("Received user message:", user_message)
        # You might want to remove this print for messages in production,
        # but keep it for now if still debugging.
        print("Received full messages object:", messages)

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen# llamaindex_pipeline.py (SUPER MINIMAL TEST - CURRENT VERSION)

        except Exception as e:
            # ... (error handling) ...
            pass