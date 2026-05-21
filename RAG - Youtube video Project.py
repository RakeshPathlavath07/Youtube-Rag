# 🎥 YouTube RAG - Retrieval Augmented Generation System

A powerful application that leverages **Retrieval Augmented Generation (RAG)** to answer questions about YouTube video transcripts. This system combines LangChain, HuggingFace models, and FAISS for intelligent transcript analysis.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running Locally with Streamlit](#running-locally-with-streamlit)
  - [Using the Python Module](#using-the-python-module)
- [How It Works](#how-it-works)
- [API Keys & Environment Variables](#api-keys--environment-variables)
- [Project Components](#project-components)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **📺 YouTube Transcript Extraction**: Automatically fetch transcripts from YouTube videos
- **🔍 Smart Search**: Uses vector embeddings to find relevant content
- **⏱️ Timestamp Preservation**: Get exact timestamps for referenced content
- **🤖 AI-Powered Answers**: Leverages Gemma 2 LLM for accurate responses
- **💬 Interactive UI**: Beautiful Streamlit interface for easy interaction
- **🎯 Context-Aware**: Answers are strictly based on video content (no hallucinations)
- **⚡ Fast Processing**: Efficient vector-based retrieval and FAISS indexing

## 🏗️ Project Structure

```
Youtube-Rag/
├── RAG - Youtube video Project.py      # Core RAG implementation
├── streamlit_app.py                    # Streamlit web interface
├── youtube_video.ipynb                 # Jupyter notebook version
├── .env.example                        # Environment variables template
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## 🔄 Architecture

```
YouTube Video
    ↓
Transcript Extraction (YouTube API)
    ↓
Text Chunking (800 chars, 200 overlap)
    ↓
Embedding Generation (MiniLM-L6-v2)
    ↓
FAISS Vector Store
    ↓
User Question
    ↓
Semantic Similarity Search (k=4)
    ↓
Context Retrieval
    ↓
Prompt Engineering
    ↓
Gemma 2 2B LLM
    ↓
Answer with Timestamps
```

## 📦 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- API Keys for:
  - HuggingFace (for model access)
  - OpenAI (optional, for embeddings)
- Internet connection for downloading models and transcripts

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/RakeshPathlavath07/Youtube-Rag.git
cd Youtube-Rag
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If `requirements.txt` doesn't exist, install manually:

```bash
pip install streamlit langchain langchain-openai langchain-huggingface langchain-community \
    youtube-transcript-api faiss-cpu sentence-transformers python-dotenv huggingface-hub
```

## ⚙️ Configuration

### Step 1: Create Environment Variables File

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Or create it manually:

```env
# .env
HUGGINGFACE_TOKEN=hf_your_huggingface_token_here
OPENAI_API_KEY=sk_your_openai_api_key_here
```

### Step 2: Get Your API Keys

#### HuggingFace Token:
1. Visit [huggingface.co](https://huggingface.co)
2. Sign up or log in
3. Go to Settings → Access Tokens
4. Create a new token with read access
5. Copy and paste it in `.env`

#### OpenAI API Key (Optional):
1. Visit [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Go to API keys section
4. Create a new API key
5. Copy and paste it in `.env`

### Step 3: Update .env with Your Keys

```env
HUGGINGFACE_TOKEN=hf_xYzAbCdEfGhIjKlMnOpQrStUvWxYz
OPENAI_API_KEY=sk_test_your_key_here_abcdefghijklmnop
```

**⚠️ Security Note**: Never commit `.env` file to version control. Add it to `.gitignore`.

## 📖 Usage

### Running Locally with Streamlit

```bash
streamlit run streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`

#### In the Streamlit Interface:

1. **Sidebar Configuration**:
   - Enter a YouTube Video ID (found in the URL: `watch?v=VIDEO_ID`)
   - Click "🚀 Load Video"

2. **Processing**:
   - Wait for transcript extraction
   - Wait for document chunking
   - Wait for embedding generation

3. **Asking Questions**:
   - Type your question in the text area
   - Click "🔍 Get Answer"
   - View the AI-generated response with timestamps

#### Example Questions:
- "What is discussed about quantization?"
- "Is the topic about plastic discussed?"
- "What are the main topics covered?"
- "Explain the techniques mentioned in this video"

### Using the Python Module

```python
from RAG___Youtube_video_Project import (
    fetch_transcript,
    create_documents_from_transcript,
    create_vector_store,
    create_retriever,
    create_prompt,
    create_rag_chain,
    answer_question,
    chat_model
)

# 1. Fetch transcript
video_id = "6S59Y0ckTm4"
transcript = fetch_transcript(video_id)

# 2. Create documents with timestamps
documents = create_documents_from_transcript(transcript)

# 3. Build vector store
vector_store = create_vector_store(documents)
retriever = create_retriever(vector_store)

# 4. Create RAG chain
prompt = create_prompt()
rag_chain = create_rag_chain(retriever, prompt, chat_model)

# 5. Ask a question
question = "What is discussed about quantization?"
answer = answer_question(rag_chain, question)
print(answer)
```

## 🔍 How It Works

### 1. **Transcript Retrieval**
- Fetches the full YouTube transcript using YouTube Transcript API
- Preserves timestamps for each spoken segment
- Handles videos with disabled captions gracefully

### 2. **Document Processing**
- Splits transcript into chunks of 800 characters with 200-character overlap
- Creates LangChain Document objects with timestamp metadata
- Ensures context continuity across chunks

### 3. **Embedding Generation**
- Uses `sentence-transformers/all-MiniLM-L6-v2` model
- Converts text chunks into 384-dimensional vectors
- Lightweight but effective for semantic similarity

### 4. **Vector Storage**
- Stores embeddings in FAISS (Facebook AI Similarity Search)
- Enables fast similarity search even with large documents
- In-memory storage for quick access

### 5. **Retrieval**
- Takes user question and converts to embedding
- Retrieves top 4 most similar document chunks
- Preserves original timestamps and context

### 6. **Generation**
- Creates a prompt with retrieved context
- Uses Gemma 2 2B-IT LLM for response generation
- Enforces strict adherence to transcript content
- Formats response with exact timestamps

## 🔐 API Keys & Environment Variables

### Supported Variables

```env
# HuggingFace Token (Required)
HUGGINGFACE_TOKEN=hf_your_token

# OpenAI API Key (Optional)
OPENAI_API_KEY=sk_your_key
```

### Loading Variables

The application uses `python-dotenv` to automatically load variables from `.env`:

```python
from dotenv import load_dotenv
import os

load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
```

### Security Best Practices

1. ✅ Store keys in `.env` file locally
2. ✅ Add `.env` to `.gitignore`
3. ✅ Use repository secrets for deployment
4. ❌ Never commit `.env` to version control
5. ❌ Don't hardcode API keys in source files
6. ❌ Don't share `.env` file via email or chat

## 🔧 Project Components

### Core Module: `RAG - Youtube video Project.py`

**Functions:**

| Function | Purpose |
|----------|---------|
| `fetch_transcript(video_id)` | Retrieves YouTube transcript with timestamps |
| `create_documents_from_transcript()` | Converts transcript into chunked LangChain documents |
| `create_vector_store(documents)` | Generates embeddings and creates FAISS index |
| `create_retriever(vector_store)` | Sets up similarity search retriever |
| `create_prompt()` | Defines the RAG prompt template |
| `create_rag_chain()` | Assembles the complete LangChain pipeline |
| `answer_question()` | Generates answer for user question |
| `format_docs()` | Formats retrieved docs with timestamps |
| `seconds_to_timestamp()` | Converts seconds to MM:SS format |

### Web Interface: `streamlit_app.py`

**Features:**
- Video ID input with examples
- Session state management for caching
- Real-time processing indicators
- Beautiful result formatting
- Error handling and user feedback
- Statistics display
- Interactive question examples

## 📚 Dependencies

```
langchain==0.1.x
langchain-openai==0.x.x
langchain-huggingface==1.x.x
langchain-community==0.x.x
langchain-text-splitters==1.x.x
youtube-transcript-api==0.6.x
faiss-cpu==1.7.x
sentence-transformers==2.x.x
huggingface-hub==0.x.x
streamlit==1.x.x
python-dotenv==1.x.x
tiktoken==0.x.x
```

Install all with:
```bash
pip install -r requirements.txt
```

## 🐛 Troubleshooting

### Issue: "No module named 'langchain'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "HUGGINGFACE_TOKEN not found"
**Solution**: 
1. Create `.env` file in project root
2. Add your HuggingFace token: `HUGGINGFACE_TOKEN=hf_...`

### Issue: "No transcripts found for this video"
**Solution**: 
- Video doesn't have captions enabled
- Try another video with English captions
- Check if video is public

### Issue: "FAISS import error"
**Solution** (CPU version):
```bash
pip install faiss-cpu
```

**Solution** (GPU version, if you have CUDA):
```bash
pip install faiss-gpu
```

### Issue: "Streamlit not launching"
**Solution**: 
```bash
streamlit run streamlit_app.py --logger.level=debug
```

### Issue: Slow embedding generation
**Solution**:
- First run downloads the model (~130MB)
- Subsequent runs are cached and faster
- Use GPU if available for faster processing

## 🔄 Workflow Example

```bash
# 1. Clone and setup
git clone https://github.com/RakeshPathlavath07/Youtube-Rag.git
cd Youtube-Rag
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
cp .env.example .env
# Edit .env and add your API keys

# 4. Run the app
streamlit run streamlit_app.py

# 5. In browser:
# - Enter video ID: 6S59Y0ckTm4
# - Click "Load Video"
# - Ask questions about the transcript
```

## 🚀 Deployment

### Streamlit Cloud

1. Fork this repository
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Create new app, connect your repo
4. Add secrets in settings:
   ```
   HUGGINGFACE_TOKEN = "hf_..."
   OPENAI_API_KEY = "sk_..."
   ```
5. Deploy!

### Docker

```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 📊 Performance

- **Transcript Processing**: ~2-5 seconds
- **Embedding Generation**: ~10-30 seconds (first run, models cached after)
- **Vector Indexing**: ~5 seconds
- **Question Answering**: ~3-8 seconds

**Optimization Tips**:
- Use GPU for faster embedding generation
- Cache vector stores for repeated queries
- Adjust chunk size for memory efficiency
- Use smaller models for faster inference

## 🎓 Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [RAG Pattern](https://js.langchain.com/docs/use_cases/question_answering)
- [FAISS Index](https://github.com/facebookresearch/faiss)
- [HuggingFace Models](https://huggingface.co/models)
- [Streamlit Docs](https://docs.streamlit.io/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Rakesh Pathlavath**
- GitHub: [@RakeshPathlavath07](https://github.com/RakeshPathlavath07)

## 🙏 Acknowledgments

- LangChain for the amazing RAG framework
- HuggingFace for open-source models
- Facebook AI for FAISS
- YouTube for the Transcript API
- Streamlit for the web framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review existing [GitHub Issues](https://github.com/RakeshPathlavath07/Youtube-Rag/issues)
3. Create a new issue with details about your problem

---

**Made with ❤️ by Rakesh Pathlavath**

⭐ If you found this helpful, please consider starring the repository!
