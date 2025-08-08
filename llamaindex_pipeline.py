"""
title: PCG RAG Pipeline
author: Your Name
date: 2025-08-08
version: 2.0
license: MIT
description: A pipeline for retrieving relevant information from a Qdrant knowledge base using LlamaIndex.
requirements: llama-index

valves:
  - name: llm_model
    type: text
    default: dolphina:latest
    description: The Ollama LLM model to use for RAG responses.
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
import traceback
import urllib.parse

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from schemas import OpenAIChatMessage  # type: ignore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama

class Pipeline:
    def __init__(self):
        self.index = None
        # Default values, will be overwritten by valves in the UI if changed
        self.llm_model = "dolphina:latest"
        self.request_timeout_seconds = 600.0
        self.top_k_retrieval = 3
        print("--- Pipeline instance initialized. ---")

    async def on_startup(self):
        print("--- on_startup method called. ---")

        qdrant_collection_name = "pcg_documents"
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "") # Workaround

        try:
            # Setup LLM from valve settings
            Settings.llm = Ollama(
                model=self.llm_model,
                base_url="http://portable-ollama:11434",
                request_timeout=self.request_timeout_seconds
            )
            print(f"--- LLM set to: {Settings.llm.model} ---")

            # Use the same HuggingFace model as the indexer
            print("--- Loading embedding model: all-MiniLM-L6-v2 ---")
            Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
            print("--- Embedding model loaded successfully. ---")

            # Connect to the Qdrant database
            print(f"--- Connecting to Qdrant collection: {qdrant_collection_name} ---")
            client = QdrantClient(host="portable-qdrant", port=6334)
            vector_store = QdrantVectorStore(client=client, collection_name=qdrant_collection_name)
            
            # Load the index directly from the Qdrant vector store
            self.index = VectorStoreIndex.from_vector_store(vector_store)
            
            print("--- Successfully connected pipeline to Qdrant index. ---")
        
        except Exception as e:
            print(f"--- CRITICAL ERROR IN ON_STARTUP: {e} ---")
            traceback.print_exc()
            raise

        print("--- on_startup completed successfully. ---")
    
    async def on_shutdown(self):
        print("--- on_shutdown called. ---")
        pass
    
    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"--- Pipe received: {user_message} ---")
        
        if not self.index:
            yield {"type": "error", "content": "Error: The Vector Index is not loaded."}
            return

        query_engine = self.index.as_query_engine(
            streaming=True, 
            similarity_top_k=int(self.top_k_retrieval)
        )
        
        response = query_engine.query(user_message)
        
        def source_generator():
            # First, stream the main text content from the LLM
            for token in response.response_gen:
                yield {"type": "stream", "content": token}

            # After the stream, prepare the structured source data
            source_nodes = response.source_nodes
            sources = []
            for node in source_nodes:
                # Corrected to use 'filename' which matches your indexer script
                filename = node.metadata.get('filename', 'Unknown Source')
                content = node.get_content()
                score = node.get_score()
                
                sources.append({
                    "name": filename,
                    "content": content,
                    "score": score,
                })

            # Yield the special "sources" message that the UI understands
            if sources:
                yield {"type": "sources", "sources": sources}

        return source_generator()
