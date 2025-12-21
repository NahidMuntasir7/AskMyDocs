from typing import List, Tuple, Dict
from sentence_transformers import CrossEncoder
from config import config


class Reranker:
    """Cross-encoder reranking for improved relevance"""
    
    def __init__(self):
        print(f"Loading reranker model: {config.RERANKER_MODEL}")
        self.model = CrossEncoder(config. RERANKER_MODEL)
        print("✅ Reranker model loaded")
    
    def rerank(
        self,
        query:  str,
        documents: List[Tuple[Dict, float]],
        top_k: int = None
    ) -> List[Tuple[Dict, float]]:
        """Rerank documents using cross-encoder"""
        if top_k is None:
            top_k = config.TOP_K_RERANK
        
        if not documents:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc['content']] for doc, _ in documents]
        
        # Get reranking scores
        scores = self.model.predict(pairs)
        
        # Combine documents with new scores
        reranked = [
            (doc, float(score))
            for (doc, _), score in zip(documents, scores)
        ]
        
        # Sort by reranking score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]