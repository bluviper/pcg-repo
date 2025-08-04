"""
title: PCG RAG Pipeline
author: Your Name
date: 2025-08-04
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index

valves:
  - name: llm_model
    type: text
    default: mistral:7b-instruct-v0.2-q4_K_M
    description: The Ollama LLM model to use for RAG responses.
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

# sys.path lines are for debugging only, remove in final
# sys.path.insert(0, '/app/backend/app')
# sys.path.insert(0, '/usr/local/lib/python3.11/site-packages/openwebui_backend/app')
# sys.path.insert(0, '/app')

from schemas import OpenAIChatMessage
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, Settings,
    StorageContext, load_index_from_storage
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None
        # Initialize attributes with their defaults, this ensures they always exist
        self.llm_model = "mistral:7b-instruct-v0.2-q4_K_M"
        self.embedding_model = "nomic-embed-text"
        self.request_timeout_seconds = 600.0
        self.top_k_retrieval = 3
        print("--- Pipeline instance initialized. ---")

    async def on_startup(self):
        print("--- on_startup method called. ---")

        PERSIST_DIR = "./index_storage"
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

        # --- Valve retrieval and LLM/Embed setup (GUARANTEED TO RUN) ---
        # Retrieve valve values. If framework doesn't set them on 'self',
        # they will fall back to the defaults initialized in __init__.
        llm_model_val = getattr(self, "llm_model", self.llm_model)
        embedding_model_val = getattr(self, "embedding_model", self.embedding_model)
        timeout_val = getattr(self, "request_timeout_seconds", self.request_timeout_seconds)
        top_k_val = getattr(self, "top_k_retrieval", self.top_k_retrieval)

        # Configure LLM and embedding models
        Settings.llm = Ollama(
            model=llm_model_val,
            base_url="http://portable-ollama:11434",
            request_timeout=timeout_val
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=embedding_model_val,
            base_url="http://portable-ollama:11434",
            request_timeout=timeout_val
        )

        print(f"--- LLM set to: {Settings.llm.model} at {Settings.llm.base_url} (Timeout: {Settings.llm.request_timeout}) ---")
        print(f"--- Embedding model: {Settings.embed_model.model_name} at {Settings.embed_model.base_url} (Timeout: {Settings.embed_model.request_timeout}) ---")

        # Debugging sys.path (optional, remove in final)
        # print("--- Current sys.path for debugging: ---")
        # for p in sys.path:
        #     print(f"  - {p}")
        # print("---------------------------------------")
        # --- End Valve retrieval and LLM/Embed setup ---

        try:
            # --- LlamaIndex indexing/loading logic (This is the part that might throw exceptions) ---
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
            # This 'except' block specifically catches errors in index creation/loading
            print(f"--- CRITICAL ERROR IN ON_STARTUP (Index/Storage): {e} ---")
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

        query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=self.top_k_retrieval)
        response = query_engine.query(user_message)

        print("--- Query executed, returning response generator. ---")
        return response.response_gen
