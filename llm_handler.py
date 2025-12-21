import requests
from typing import List, Dict
from config import config
from utils import count_tokens, truncate_memory


class LLMHandler:
    """Handle LLM requests to GitHub Models API with conversational memory"""
    
    def __init__(self):
        self.api_base = config. GITHUB_API_BASE
        self.model = config.MODEL_NAME
        self.headers = {
            "Authorization": f"Bearer {config.GITHUB_TOKEN}",
            "Content-Type": "application/json"
        }
    
    def generate_answer(
        self,
        query: str,
        context_docs: List[Dict],
        chat_history: List[Dict] = None
    ) -> Dict[str, any]:
        """Generate answer using RAG with conversational memory"""
        
        # Prepare context from documents
        context = self._format_context(context_docs)
        
        # Prepare chat history
        if chat_history is None:
            chat_history = []
        
        # Truncate history to fit token limit
        truncated_history = truncate_memory(
            chat_history[-config.MEMORY_WINDOW: ],
            config.MAX_MEMORY_TOKENS
        )
        
        # Create prompt with memory
        messages = self._create_messages_with_memory(query, context, truncated_history)
        
        # Call API
        response = self._call_api(messages)
        
        return {
            'answer': response,
            'sources': context_docs
        }
    
    def _format_context(self, docs: List[Dict]) -> str:
        """Format retrieved documents as context"""
        context_parts = []
        for i, (doc, score) in enumerate(docs, 1):
            context_parts.append(
                f"[Document {i}] (Relevance: {score:.3f})\n"
                f"{doc['content']}\n"
                f"Source: {doc['metadata']['filename']}, Page {doc['metadata']['page']}\n"
            )
        return "\n". join(context_parts)
    
    def _create_messages_with_memory(
        self,
        query: str,
        context: str,
        chat_history: List[Dict]
    ) -> List[Dict]:
        """Create messages array with system prompt, history, and current query"""
        
        messages = [
            {
                "role": "system",
                "content": """You are an AI assistant that answers questions based on provided documents. 

                Instructions:
                1. Answer questions using ONLY the information from the provided documents
                2. If the answer is not in the documents, say "I cannot find this information in the provided documents"
                3. Use the conversation history to understand context and follow-up questions
                4. Be concise and accurate
                5. Cite specific document numbers when relevant [Document X]
                6. Handle pronouns and references based on conversation history (e.g., "it", "that", "what you said")"""
            }
        ]
        
        # Add chat history (previous Q&A pairs)
        for item in chat_history:
            messages.append({
                "role": "user",
                "content": item['query']
            })
            messages. append({
                "role": "assistant",
                "content": item['answer']
            })
        
        # Add current query with documents
        current_prompt = f"""Documents: 
{context}

Question: {query}

Answer:"""
        
        messages. append({
            "role": "user",
            "content": current_prompt
        })
        
        return messages
    
    def _call_api(self, messages:  List[Dict]) -> str:
        """Call GitHub Models API"""
        url = f"{self.api_base}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages":  messages,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS
        }
        
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"Error calling LLM API: {str(e)}"