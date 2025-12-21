import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config: 
    """Configuration for RAG system"""
    
    # GitHub Models API
    GITHUB_TOKEN:  str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_API_BASE: str = "https://models.github.ai/inference"
    MODEL_NAME: str = "gpt-4o"
    
    # Embedding Model
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIM: int = 768
    
    # Reranker Model
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Document Processing
    CHUNK_SIZE: int = 500  # tokens
    CHUNK_OVERLAP:  int = 50
    
    # Retrieval
    TOP_K_RETRIEVAL: int = 20  # Initial retrieval
    TOP_K_RERANK: int = 5  # After reranking
    
    # BM25 Weight (0.0 = only semantic, 1.0 = only BM25)
    BM25_WEIGHT: float = 0.3
    
    # LLM Parameters
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    
    # Memory Configuration
    MEMORY_WINDOW: int = 5  # Number of previous Q&A pairs to remember
    MAX_MEMORY_TOKENS: int = 2000  # Max tokens for memory context
    
    # Storage
    VECTOR_STORE_PATH: str = "./vector_store"
    UPLOAD_DIR: str = "./uploads"
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.VECTOR_STORE_PATH, exist_ok=True)
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        
        if not self.GITHUB_TOKEN: 
            raise ValueError(
                "GITHUB_TOKEN not found. Please create a .env file with:\n"
                "GITHUB_TOKEN=your_token_here\n\n"
                "Get your token from:  https://github.com/settings/tokens"
            )


config = Config()