import re
import tiktoken
from typing import List


def count_tokens(text: str, model:  str = "gpt-4") -> int:
    """Count tokens in text"""
    try:
        encoding = tiktoken. encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = re. sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']', '', text)
    return text.strip()


def split_text_by_tokens(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """Split text into chunks by token count with overlap"""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap
        
    return chunks


def format_sources(sources: List[dict]) -> str:
    """Format source citations"""
    formatted = []
    for i, source in enumerate(sources, 1):
        formatted.append(
            f"**Source {i}** (Score: {source['score']:.3f})\n"
            f"📄 {source['filename']} - Page {source['page']}\n"
            f"```\n{source['content'][: 300]}...\n```"
        )
    return "\n\n".join(formatted)


def truncate_memory(memory:  List[dict], max_tokens:  int) -> List[dict]:
    """Truncate memory to fit within token limit"""
    truncated = []
    total_tokens = 0
    
    # Add from most recent to oldest
    for item in reversed(memory):
        item_tokens = count_tokens(item['query'] + item['answer'])
        if total_tokens + item_tokens > max_tokens:
            break
        truncated.insert(0, item)
        total_tokens += item_tokens
    
    return truncated