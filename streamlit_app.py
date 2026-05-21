import streamlit as st
import os
from dotenv import load_dotenv
from RAG___Youtube_video_Project import (
    fetch_transcript,
    create_documents_from_transcript,
    create_vector_store,
    create_retriever,
    create_prompt,
    create_rag_chain,
    answer_question,
    seconds_to_timestamp,
    chat_model
)

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="YouTube Video RAG",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🎥 YouTube Video RAG System")
st.markdown("""
This application uses Retrieval Augmented Generation (RAG) to answer questions about YouTube video content.
Simply provide a YouTube video ID and ask questions about the video's transcript!
""")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Video ID input
video_id = st.sidebar.text_input(
    "Enter YouTube Video ID",
    value="6S59Y0ckTm4",
    help="The unique identifier found in YouTube URLs (e.g., 'watch?v=VIDEO_ID')"
)

# Session state for caching
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

# Load button
col1, col2 = st.sidebar.columns(2)

with col1:
    load_button = st.button("🚀 Load Video", use_container_width=True)

with col2:
    clear_button = st.button("🗑️ Clear Cache", use_container_width=True)

if clear_button:
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.rag_chain = None
    st.session_state.current_video_id = None
    st.success("Cache cleared!")

# Load video and create RAG chain
if load_button or (st.session_state.current_video_id == video_id and st.session_state.rag_chain):
    if video_id:
        with st.spinner("📥 Fetching transcript..."):
            try:
                transcript = fetch_transcript(video_id)
                
                if transcript:
                    with st.spinner("📄 Creating documents..."):
                        documents = create_documents_from_transcript(transcript)
                    
                    with st.spinner("🔢 Building embeddings..."):
                        st.session_state.vector_store = create_vector_store(documents)
                        st.session_state.retriever = create_retriever(st.session_state.vector_store)
                    
                    with st.spinner("⛓️ Creating RAG chain..."):
                        prompt = create_prompt()
                        st.session_state.rag_chain = create_rag_chain(
                            st.session_state.retriever,
                            prompt,
                            chat_model
                        )
                    
                    st.session_state.current_video_id = video_id
                    st.success(f"✅ Successfully loaded video! ({len(documents)} chunks created)")
                    
                    # Display transcript statistics
                    st.sidebar.metric("Total Chunks", len(documents))
                else:
                    st.error("❌ Failed to fetch transcript. Check the video ID and try again.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("Please enter a video ID")

# Main content area
if st.session_state.rag_chain:
    st.success("✅ Video loaded! You can now ask questions about the transcript.")
    
    # Question input
    st.markdown("---")
    st.subheader("❓ Ask a Question")
    
    question = st.text_area(
        "What would you like to know about this video?",
        placeholder="e.g., What is discussed about quantization?",
        height=100
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ask_button = st.button("🔍 Get Answer", use_container_width=True)
    
    with col2:
        example_button = st.button("💡 Example Question", use_container_width=True)
    
    with col3:
        clear_question = st.button("🔄 Clear", use_container_width=True)
    
    if example_button:
        question = "What are the main topics discussed in this video?"
        st.rerun()
    
    if clear_question:
        st.rerun()
    
    # Process question
    if ask_button:
        if question.strip():
            with st.spinner("🤔 Analyzing transcript..."):
                try:
                    answer = answer_question(st.session_state.rag_chain, question)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📝 Answer")
                    
                    # Create two columns for better layout
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(answer)
                    
                    with col2:
                        st.info(
                            """
                            **Note:** Answers are based strictly on the video transcript.
                            Timestamps are provided where relevant topics are discussed.
                            """
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error processing question: {str(e)}")
        else:
            st.warning("Please enter a question!")

else:
    # Initial state
    st.info(
        """
        👈 **To get started:**
        1. Enter a YouTube Video ID in the sidebar
        2. Click "Load Video" to fetch and process the transcript
        3. Ask questions about the video content
        """
    )
    
    st.markdown("---")
    st.subheader("📖 How it Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Step 1: Transcript Retrieval**
        - Fetches the transcript from YouTube
        - Preserves timestamps for each segment
        
        **Step 2: Document Processing**
        - Splits transcript into chunks (800 chars with overlap)
        - Creates document metadata with timestamps
        """)
    
    with col2:
        st.markdown("""
        **Step 3: Vector Store Creation**
        - Generates embeddings using HuggingFace models
        - Stores embeddings in FAISS vector database
        
        **Step 4: RAG Chain**
        - Retrieves relevant chunks for your question
        - Uses Gemma 2 LLM to generate accurate answers
        - Includes timestamp references
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("🛠️ **Tech Stack**")
    st.caption("LangChain • HuggingFace • FAISS • Streamlit")

with col2:
    st.markdown("📚 **Models Used**")
    st.caption("Gemma 2 2B • MiniLM Embeddings")

with col3:
    st.markdown("🔗 **APIs**")
    st.caption("YouTube Transcript API")
