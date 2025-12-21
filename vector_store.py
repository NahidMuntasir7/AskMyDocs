import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Tuple
from config import config


class VectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(self):
        self.index = None
        self.documents = []
        self.index_path = os.path.join(config. VECTOR_STORE_PATH, "faiss. index")
        self.docs_path = os.path.join(config.VECTOR_STORE_PATH, "documents. pkl")
    
    def create_index(self, embeddings: np.ndarray, documents: List[Dict]):
        """Create FAISS index from embeddings"""
        dimension = embeddings.shape[1]
        
        # Use IndexFlatL2 for exact search
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings. astype('float32'))
        self.documents = documents
        
        print(f"✅ Created FAISS index with {len(documents)} documents")
    
    def save(self):
        """Save index and documents to disk"""
        if self.index is not None:
            faiss.write_index(self. index, self.index_path)
            with open(self.docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            print("✅ Vector store saved")
    
    def load(self) -> bool:
        """Load index and documents from disk"""
        if os. path.exists(self.index_path) and os.path.exists(self.docs_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"✅ Loaded vector store with {len(self.documents)} documents")
            return True
        return False
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Dict, float]]:
        """Search for similar documents"""
        if self.index is None:
            raise ValueError("Index not initialized")
        
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                # Convert L2 distance to similarity score
                similarity = 1 / (1 + distance)
                results.append((self.documents[idx], similarity))
        
        return results
    
    def clear(self):
        """Clear the index"""
        self.index = None
        self. documents = []
        if os.path.exists(self. index_path):
            os.remove(self.index_path)
        if os.path.exists(self.docs_path):
            os.remove(self.docs_path)
        print("✅ Vector store cleared")