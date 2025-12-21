import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from config import config


class EmbeddingManager:
    """Generate embeddings using local transformer models"""
    
    def __init__(self):
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        print("✅ Embedding model loaded")
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for documents"""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True  # For cosine similarity
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query"""
        # Add instruction for better retrieval (BGE models)
        instructed_query = f"Represent this sentence for searching relevant passages: {query}"
        embedding = self.model.encode(
            instructed_query,
            normalize_embeddings=True
        )
        return embedding