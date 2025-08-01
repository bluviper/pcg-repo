"""
title: Llama Index RAG Pipeline # This title will appear in the UI
author: Your Name
date: 2025-07-31
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index
"""

from typing import List, Union, Generator, Iterator

# LlamaIndex imports
import os
import sys # Keep sys and os if you use sys.path.insert for other debugging, otherwise can remove if not needed.

# --- REMOVE THESE sys.path DEBUGGING LINES FOR GITHUB UPLOAD ---
# They are not needed when the file is directly managed by the server's loader.
# sys.path.insert(0, '/app/backend/app')
# sys.path.insert(0, '/usr/local/lib/python3.11/site-packages/openwebui_backend/app')
# sys.path.insert(0, '/app')
# --- END sys.path DEBUGGING ---

from schemas import OpenAIChatMessage # Keep UNCOMMENTED
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# If you plan to use Qdrant directly as the vector store for LlamaIndex, uncomment these:
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient

# --- Remove these global prints from final version ---
# print("--- Llama Index RAG Pipeline script started. ---")

class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None
        # print("--- Pipeline instance initialized (in __init__). ---") # Remove from final

    async def on_startup(self):
        # print("--- on_startup method called. ---") # Remove from final

        # Define the persistent storage path inside the container
        PERSIST_DIR = "./index_storage" # Maps to your host's ./index_storage

        # Ensure this API key is consistent (from docker-compose env)
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "") # Set default to empty string if not explicitly needed
        # print(f"--- OPENAI_API_KEY set: {os.environ['OPENAI_API_KEY']} ---") # Remove from final

        try:
            # Configure LlamaIndex to use Ollama for LLM
            # IMPORTANT: Ollama (LLM) expects 'model='
            Settings.llm = Ollama(model="mistral:7b-instruct-v0.2-q4_K_M", base_url="http://portable-ollama:11434")
            # print(f"--- LlamaIndex LLM set to Ollama: {Settings.llm.model} at {Settings.llm.base_url}. ---") # Remove from final

            # Configure LlamaIndex to use Ollama for Embeddings
            # IMPORTANT: OllamaEmbedding expects 'model_name='
            Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://portable-ollama:11434")
            # print(f"--- LlamaIndex Embed Model set to Ollama: {Settings.embed_model.model_name} at {Settings.embed_model.base_url}. ---") # Remove from final
            
            # --- Persistence Logic ---
            if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
                # print(f"--- Index storage directory {PERSIST_DIR} not found or empty. Building new index. ---") # Remove from final
                self.documents = SimpleDirectoryReader("./docs").load_data()
                # print(f"--- Loaded {len(self.documents)} documents from ./docs. ---") # Remove from final
                
                self.index = VectorStoreIndex.from_documents(self.documents)
                # print("--- LlamaIndex vector store index created successfully. ---") # Remove from final
                
                self.index.storage_context.persist(persist_dir=PERSIST_DIR)
                # print(f"--- Index persisted to {PERSIST_DIR}. ---") # Remove from final
            else:
                # print(f"--- Loading index from persistent storage: {PERSIST_DIR}. ---") # Remove from final
                self.index = load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))
                # print("--- LlamaIndex vector store index loaded from storage. ---") # Remove from final

        except Exception as e:
            # Keep robust error logging for debugging, or replace with a proper logger
            print(f"--- CRITICAL ERROR IN ON_STARTUP: {e} ---") 
            import traceback
            traceback.print_exc()
            raise # Re-raise to ensure the server process crashes if startup fails

        # print("--- on_startup method finished successfully. ---") # Remove from final
        pass

    async def on_shutdown(self):
        # print("--- on_shutdown method called. ---") # Remove from final
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # print("--- pipe method called. ---") # Remove from final
        # print("Received user message:", user_message) # Remove from final
        # print("Received full messages object:", messages) # Remove from final

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)
        # print("--- Query executed, returning response generator. ---") # Remove from final

        return response.response_gen
