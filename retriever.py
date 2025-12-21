from typing import List, Tuple, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from embedding_manager import EmbeddingManager
from vector_store import VectorStore
from config import config


class HybridRetriever:
    """Hybrid retrieval combining dense (semantic) and sparse (BM25) search"""
    
    def __init__(self, embedding_manager: EmbeddingManager, vector_store: VectorStore):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.bm25 = None
        self._init_bm25()
    
    def _init_bm25(self):
        """Initialize BM25 index"""
        if not self.vector_store.documents:
            return
        
        tokenized_corpus = [
            doc['content']. lower().split()
            for doc in self.vector_store.documents
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("✅ BM25 index initialized")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[Dict, float]]:
        """Hybrid retrieval combining semantic and BM25 search"""
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL
        
        # 1. Semantic search (dense)
        query_embedding = self.embedding_manager.embed_query(query)
        semantic_results = self.vector_store.search(query_embedding, k=top_k)
        
        # 2. BM25 search (sparse)
        if self.bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k BM25 results
            top_indices = np.argsort(bm25_scores)[::-1][:top_k]
            bm25_results = [
                (self.vector_store.documents[idx], bm25_scores[idx])
                for idx in top_indices
            ]
        else:
            bm25_results = []
        
        # 3. Combine scores using weighted fusion
        combined = self._combine_results(
            semantic_results,
            bm25_results,
            alpha=config.BM25_WEIGHT
        )
        
        return combined[: top_k]
    
    def _combine_results(
        self,
        semantic_results: List[Tuple[Dict, float]],
        bm25_results: List[Tuple[Dict, float]],
        alpha: float = 0.3
    ) -> List[Tuple[Dict, float]]:
        """Combine semantic and BM25 scores"""
        # Normalize scores
        semantic_scores = self._normalize_scores([s for _, s in semantic_results])
        bm25_scores = self._normalize_scores([s for _, s in bm25_results])
        
        # Create score dictionary
        score_dict = {}
        
        for (doc, _), norm_score in zip(semantic_results, semantic_scores):
            doc_id = doc['metadata']['chunk_id']
            score_dict[doc_id] = {
                'doc':  doc,
                'score': (1 - alpha) * norm_score
            }
        
        for (doc, _), norm_score in zip(bm25_results, bm25_scores):
            doc_id = doc['metadata']['chunk_id']
            if doc_id in score_dict: 
                score_dict[doc_id]['score'] += alpha * norm_score
            else: 
                score_dict[doc_id] = {
                    'doc': doc,
                    'score': alpha * norm_score
                }
        
        # Sort by combined score
        combined = [
            (item['doc'], item['score'])
            for item in score_dict.values()
        ]
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return combined
    
    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1]"""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score: 
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]