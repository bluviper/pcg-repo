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
        self.llm_model = "mistral:7b-instruct-v0.2-q4_K_M"
        self.embedding_model = "nomic-embed-text"
        self.request_timeout_seconds = 600.0
        self.top_k_retrieval = 3
        print("--- Pipeline instance initialized. ---")

    async def on_startup(self):
        print("--- on_startup method called. ---")

        PERSIST_DIR = "./index_storage"
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

        try:
            llm_model_val = getattr(self, "llm_model", self.llm_model)
            embedding_model_val = getattr(self, "embedding_model", self.embedding_model)
            timeout_val = getattr(self, "request_timeout_seconds", self.request_timeout_seconds)
            top_k_val = getattr(self, "top_k_retrieval", self.top_k_retrieval)

            Settings.llm = Ollama(
                model=llm_model_val,
                base_url="http://portable-ollama:11434",
                request_timeout=timeout_val
            )
            print(f"--- LLM set to: {Settings.llm.model} at {Settings.llm.base_url} (Timeout: {Settings.llm.request_timeout}) ---")

            print("--- Attempting to set Embedding Model. ---")
            Settings.embed_model = OllamaEmbedding(
                model_name=embedding_model_val,
                base_url="http://portable-ollama:11434",
                request_timeout=timeout_val
            )
            print(f"--- Embedding model: {Settings.embed_model.model_name} at {Settings.embed_model.base_url} ---") # CORRECTED LINE: Removed .request_timeout

            print("--- Current sys.path for debugging: ---")
            for p in sys.path:
                print(f"  - {p}")
            print("---------------------------------------")

            print("--- Starting index creation/loading logic. ---")
            if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
                print(f"--- Index storage directory {PERSIST_DIR} not found or empty. Building new index. ---")
                self.documents = SimpleDirectoryReader("./docs").load_data()
                print(f"--- Loaded {len(self.documents)} documents from ./docs. ---")
                
                print("--- Attempting to create VectorStoreIndex from documents. ---")
                self.index = VectorStoreIndex.from_documents(self.documents)
                print("--- VectorStoreIndex creation successful. ---")
                
                print("--- Attempting to persist index. ---")
                self.index.storage_context.persist(persist_dir=PERSIST_DIR)
                print(f"--- Index persisted to {PERSIST_DIR}. ---")
            else:
                print(f"--- Loading index from {PERSIST_DIR}. ---")
                print("--- Attempting to load VectorStoreIndex from storage. ---")
                self.index = load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))
                print("--- VectorStoreIndex loaded from storage successful. ---")
            print("--- Finished index creation/loading logic. ---")
        
        except Exception as e:
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
