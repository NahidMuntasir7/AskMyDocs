import os
from typing import List, Dict
import PyPDF2
import pdfplumber
from docx import Document
from utils import clean_text, split_text_by_tokens
from config import config


class DocumentProcessor:
    """Extract and process text from various document formats"""
    
    def __init__(self):
        self.supported_formats = ['. pdf', '.docx', '.txt']
    
    def process_file(self, file_path: str) -> List[Dict[str, any]]:
        """Process a single file and return chunks with metadata"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            text_by_page = self._extract_pdf(file_path)
        elif ext == '.docx':
            text_by_page = self._extract_docx(file_path)
        elif ext == '.txt': 
            text_by_page = self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Create chunks with metadata
        chunks = []
        for page_num, text in text_by_page. items():
            cleaned = clean_text(text)
            if not cleaned:
                continue
                
            page_chunks = split_text_by_tokens(
                cleaned,
                chunk_size=config. CHUNK_SIZE,
                overlap=config.CHUNK_OVERLAP
            )
            
            for chunk_idx, chunk in enumerate(page_chunks):
                chunks.append({
                    'content': chunk,
                    'metadata': {
                        'filename': os.path.basename(file_path),
                        'page': page_num,
                        'chunk_id': f"{page_num}_{chunk_idx}",
                        'source': file_path
                    }
                })
        
        return chunks
    
    def _extract_pdf(self, file_path: str) -> Dict[int, str]:
        """Extract text from PDF"""
        text_by_page = {}
        
        try:
            # Try pdfplumber first (better for complex PDFs)
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_by_page[i] = text
        except Exception: 
            # Fallback to PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text: 
                        text_by_page[i] = text
        
        return text_by_page
    
    def _extract_docx(self, file_path: str) -> Dict[int, str]:
        """Extract text from DOCX"""
        doc = Document(file_path)
        text_by_page = {}
        
        full_text = []
        for para in doc.paragraphs:
            if para.text. strip():
                full_text. append(para.text)
        
        # DOCX doesn't have pages, so we simulate pages (every 500 words)
        all_text = '\n'.join(full_text)
        words = all_text.split()
        
        words_per_page = 500
        for i in range(0, len(words), words_per_page):
            page_num = (i // words_per_page) + 1
            page_text = ' '.join(words[i:i + words_per_page])
            text_by_page[page_num] = page_text
        
        return text_by_page
    
    def _extract_txt(self, file_path:  str) -> Dict[int, str]:
        """Extract text from TXT"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        # Split into pseudo-pages
        words = text.split()
        text_by_page = {}
        words_per_page = 500
        
        for i in range(0, len(words), words_per_page):
            page_num = (i // words_per_page) + 1
            page_text = ' '.join(words[i:i + words_per_page])
            text_by_page[page_num] = page_text
        
        return text_by_page