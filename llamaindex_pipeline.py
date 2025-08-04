"""
title: PCG RAG Pipeline
author: Your Name
date: 2025-08-04
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index

valves: # No indentation before 'valves:'
  - name: llm_model # Two spaces indent
    type: text # Two spaces indent
    default: mistral:7b-instruct-v0.2-q4_K_M # Two spaces indent
    description: The Ollama LLM model to use for RAG responses. # Two spaces indent
  - name: embedding_model
    type: text
    default: nomic-embed-text
    description: The Ollama embedding model for indexing and retrieval.
  - name: request_timeout_seconds
    type: number
    default: 600.0
    description: Timeout for Ollama requests in seconds.
  - name: top_k_retrieval
    type: number
    default: 3
    description: Number of top relevant documents to retrieve for RAG.
"""




from typing import List, Union, Generator, Iterator
import sys
import os

# --- ADD THESE LINES FOR DEBUGGING sys.path ---
sys.path.insert(0, '/app/backend/app')
sys.path.insert(0, '/usr/local/lib/python3.11/site-packages/openwebui_backend/app')
sys.path.insert(0, '/app')
# --- END DEBUGGING sys.path ---

from schemas import OpenAIChatMessage
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, Settings,
    StorageContext, load_index_from_storage
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# If using Qdrant in the future:
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient

print("--- Llama Index RAG Pipeline script started. ---")

class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None
        print("--- Pipeline instance initialized. ---")

    async def on_startup(self):
        print("--- on_startup method called. ---")

        PERSIST_DIR = "./index_storage"
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

        # IMPORTANT: Retrieve valve values from self (the Pipeline instance)
        # The Pipelines framework will inject these from the UI.
        llm_model = getattr(self, "llm_model", "mistral:7b-instruct-v0.2-q4_K_M") # Default if not set
        embedding_model = getattr(self, "embedding_model", "nomic-embed-text")
        timeout = getattr(self, "request_timeout_seconds", 600.0)
        top_k = getattr(self, "top_k_retrieval", 3)
      
        try:
            # Configure LLM and embedding
            Settings.llm = Ollama(
                model="mistral:7b-instruct-v0.2-q4_K_M",
                base_url="http://portable-ollama:11434",
                request_timeout=timeout
            )
            Settings.embed_model = OllamaEmbedding(
                model_name="nomic-embed-text",
                base_url="http://portable-ollama:11434",
                request_timeout=timeout
            )

            print(f"--- LLM set to: {Settings.llm.model} at {Settings.llm.base_url}")
            print(f"--- Embedding model: {Settings.embed_model.model_name} at {Settings.embed_model.base_url}")

            print("--- Current sys.path for debugging: ---")
            for p in sys.path:
                print(f"  - {p}")
            print("---------------------------------------")

            if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
                print(f"--- Index storage directory {PERSIST_DIR} not found or empty. Building new index. ---")
                self.documents = SimpleDirectoryReader("./docs").load_data()
                print(f"--- Loaded {len(self.documents)} documents from ./docs. ---")
                self.index = VectorStoreIndex.from_documents(self.documents)
                print("--- LlamaIndex vector store index created. ---")
                self.index.storage_context.persist(persist_dir=PERSIST_DIR)
                print(f"--- Index persisted to {PERSIST_DIR}. ---")
            else:
                print(f"--- Loading index from {PERSIST_DIR}. ---")
                self.index = load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))
                print("--- LlamaIndex index loaded from storage. ---")

        except Exception as e:
            print(f"--- CRITICAL ERROR IN ON_STARTUP: {e} ---")
            import traceback
            traceback.print_exc()
            raise

        print("--- on_startup completed successfully. ---")

    async def on_shutdown(self):
        print("--- on_shutdown called. ---")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print("--- pipe method called. ---")
        print("Received user message:", user_message)
        print("Received full messages object:", messages)

        query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=top_k)
        response = query_engine.query(user_message)

        print("--- Query executed, returning response generator. ---")
        return response.response_gen
