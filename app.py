import streamlit as st
import os
import time
from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager
from vector_store import VectorStore
from retriever import HybridRetriever
from reranker import Reranker
from llm_handler import LLMHandler
from config import config
from utils import format_sources


# Page config
st.set_page_config(
    page_title="AI Document Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .source-box {
        background-color: #37474f;  
        color: white;  
        padding: 1rem;
        border-radius: 0.5rem;
        margin:  0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #1976d2;    
        border-left: 4px solid #0d47a1;
        color: white;  
    }
    .assistant-message {
        background-color: #7b1fa2;  
        border-left: 4px solid #4a0072;
        color: white;
    }
    .memory-badge {
        display: inline-block;
        background:  linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size:  0.8rem;
        margin-left: 0.5rem;
    }
}
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st. session_state.vector_store = None
    st. session_state.retriever = None
    st.session_state. reranker = None
    st.session_state.llm = None
    st.session_state.embedding_manager = None
    st. session_state. doc_processor = None
    st. session_state.chat_history = []


def initialize_system():
    """Initialize all components"""
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing AI system..."):
            try:
                st.session_state.embedding_manager = EmbeddingManager()
                st.session_state.vector_store = VectorStore()
                st.session_state. doc_processor = DocumentProcessor()
                st.session_state.reranker = Reranker()
                st.session_state. llm = LLMHandler()
                
                # Try to load existing index
                st.session_state.vector_store.load()
                
                st.session_state.retriever = HybridRetriever(
                    st.session_state.embedding_manager,
                    st.session_state.vector_store
                )
                
                st.session_state.initialized = True
                st.success("✅ System initialized successfully!")
            except Exception as e:
                st.error(f"❌ Initialization error: {str(e)}")
                st.stop()


def process_uploaded_files(files):
    """Process uploaded files and create vector index"""
    all_chunks = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Save uploaded file
        file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file. getbuffer())
        
        # Process document
        try:
            chunks = st.session_state. doc_processor.process_file(file_path)
            all_chunks.extend(chunks)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}:  {str(e)}")
        
        progress_bar.progress((idx + 1) / len(files))
    
    status_text.text("Creating embeddings...")
    
    # Generate embeddings
    texts = [chunk['content'] for chunk in all_chunks]
    embeddings = st.session_state. embedding_manager.embed_documents(texts)
    
    # Create vector store
    st.session_state.vector_store.create_index(embeddings, all_chunks)
    st.session_state.vector_store.save()
    
    # Reinitialize retriever
    st. session_state.retriever = HybridRetriever(
        st.session_state.embedding_manager,
        st.session_state.vector_store
    )
    
    progress_bar.progress(100)
    status_text.text("✅ Processing complete!")
    
    return len(all_chunks)


def answer_question(query:  str):
    """Answer question using RAG pipeline with memory"""
    
    with st.spinner("🔍 Searching documents..."):
        # Retrieve
        retrieved_docs = st.session_state.retriever.retrieve(query)
    
    with st.spinner("🎯 Reranking results..."):
        # Rerank
        reranked_docs = st.session_state.reranker.rerank(query, retrieved_docs)
    
    with st.spinner("🤖 Generating answer with memory..."):
        # Generate answer with chat history
        result = st.session_state.llm.generate_answer(
            query,
            reranked_docs,
            chat_history=st.session_state.chat_history
        )
    
    return result


# Main UI
def main():
    st.markdown('<h1 class="main-header">🤖 AI Document Q&A System</h1>', unsafe_allow_html=True)
    st.markdown("""
    ### Advanced RAG with Hybrid Search, Reranking & Conversational Memory
    <span class="memory-badge">Memory Enabled</span>
    """, unsafe_allow_html=True)
    
    initialize_system()
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Document Management")
        
        # Upload section
        uploaded_files = st.file_uploader(
            "Upload Document",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files"
        )
       
        
        if uploaded_files:
            if st.button("🚀 Process Documents"):
                num_chunks = process_uploaded_files(uploaded_files)
                st.success(f"✅ Processed {len(uploaded_files)} files into {num_chunks} chunks")
        
        st.divider()
        
        # Stats
        if st.session_state.vector_store and st.session_state.vector_store.documents:
            st.metric("Total Chunks", len(st.session_state.vector_store.documents))
            
            unique_files = set(
                doc['metadata']['filename']
                for doc in st.session_state.vector_store.documents
            )
            st.metric("Documents Indexed", len(unique_files))
            st.metric("💭 Conversations", len(st. session_state.chat_history))
            st.metric("🧠 Memory Window", f"{config.MEMORY_WINDOW} turns")
        
        st.divider()
        
        # Clear actions
        col1, col2 = st. columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state. chat_history = []
                st.rerun()
        with col2:
            if st. button("🗑️ Clear Index"):
                st.session_state.vector_store.clear()
                st.session_state.chat_history = []
                st. rerun()
        
        st.divider()
        
        # Settings
        with st.expander("⚙️ Advanced Settings"):
            config.TOP_K_RETRIEVAL = st.slider("Initial Retrieval", 5, 50, 20)
            config.TOP_K_RERANK = st.slider("Final Results", 3, 10, 5)
            config.BM25_WEIGHT = st.slider("BM25 Weight", 0.0, 1.0, 0.3)
            config.LLM_TEMPERATURE = st. slider("Temperature", 0.0, 1.0, 0.1)
            config.MEMORY_WINDOW = st.slider("Memory Window", 1, 10, 5)
    
    # Main area
    if not st.session_state.vector_store or not st.session_state.vector_store.documents:
        st.info("👈 Please upload and process documents to get started")
        
        st.markdown("""
        ### ✨ Features
        - 📄 **Multi-format Support**: PDF, DOCX, TXT
        - 🔍 **Hybrid Search**: Semantic + BM25 retrieval
        - 🎯 **Reranking**: Cross-encoder for accuracy
        - 🧠 **Conversational Memory**: Remembers context across questions
        - 🤖 **GPT-4o**: Powered by GitHub Models API
        - 📊 **Source Attribution**: View relevant passages
        
        ### 💬 Example Follow-up Questions
        - "What about X?" (refers to previous context)
        - "Can you explain that more?"
        - "Tell me more about what you just said"
        - "How does that relate to Y?"
        """)
    else:
        # Chat interface
        st.subheader("💬 Ask Questions")
        
        query = st.text_input(
            "Your Question:",
            placeholder="What is this document about?",
            key="query_input"
        )
        
        col1, col2 = st. columns([1, 4])
        with col1:
            ask_button = st.button("🔍 Ask")
        
        if ask_button and query:
            # Answer question with memory
            result = answer_question(query)
            
            # Add to history
            st.session_state.chat_history.append({
                'query': query,
                'answer': result['answer'],
                'sources': result['sources']
            })
            
            # Clear input
            st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("💭 Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                turn_num = len(st.session_state.chat_history) - i
                
                # User message
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🙋 Question #{turn_num}:</strong> {chat['query']}
                </div>
                """, unsafe_allow_html=True)
                
                # Assistant message
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Answer:</strong><br><br>{chat['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Sources
                with st.expander(f"📚 View Sources for Question #{turn_num}"):
                    for j, (doc, score) in enumerate(chat['sources'], 1):
                        st.markdown(f"""
                        <div class="source-box">
                        <strong>Source {j}</strong> (Relevance: {score:.3f})<br>
                        📄 {doc['metadata']['filename']} - Page {doc['metadata']['page']}<br>
                        <pre>{doc['content'][: 300]}...</pre>
                        </div>
                        """, unsafe_allow_html=True)
                
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("---")


if __name__ == "__main__":
    main()