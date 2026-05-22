import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get HuggingFace API Token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "your_huggingface_token_here")
# Langchain requires the token to be in this specific environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_TOKEN

"""## Install libraries"""

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Using Hugging Face purely for the LLM (google/gemma-2-2b-it)
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
)
chat_model = ChatHuggingFace(llm=llm)

"""## Step 1a - Indexing (Document Ingestion)"""

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

def fetch_transcript(video_id):
    """Fetch transcript from YouTube video"""
    api = YouTubeTranscriptApi()
    
    try:
        transcript_list = api.fetch(video_id, languages=['en'])

        # ✅ Timestamp-aware transcript storage
        transcript_with_timestamps = []

        for chunk in transcript_list:
            transcript_with_timestamps.append({
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.start + chunk.duration
            })

        # Optional: print sample
        for item in transcript_with_timestamps[:10]:
            print(
                f"[{item['start']:.2f}s → {item['end']:.2f}s] {item['text']}"
            )
        
        return transcript_with_timestamps

    except TranscriptsDisabled:
        print("No captions available for this video.")
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

"""## Step 1b - Indexing (Text Splitting)"""

def seconds_to_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

from langchain_core.documents import Document

CHUNK_CHAR_LIMIT = 800
CHUNK_OVERLAP = 200

def create_documents_from_transcript(transcript_with_timestamps):
    """Create documents with timestamps from transcript"""
    documents = []

    current_text = ""
    current_start = None

    for item in transcript_with_timestamps:
        # initialize start time for a new chunk
        if current_start is None:
            current_start = item["start"]

        current_text += " " + item["text"]

        # when chunk size reached → create document
        if len(current_text) >= CHUNK_CHAR_LIMIT:
            documents.append(
                Document(
                    page_content=current_text.strip(),
                    metadata={
                        "start": current_start,
                        "end": item["end"]
                    }
                )
            )

            # keep overlap text for next chunk
            current_text = current_text[-CHUNK_OVERLAP:]
            current_start = item["start"]

    # handle remaining text
    if current_text.strip():
        documents.append(
            Document(
                page_content=current_text.strip(),
                metadata={
                    "start": current_start,
                    "end": transcript_with_timestamps[-1]["end"]
                }
            )
        )

    print(f"Total chunks created: {len(documents)}")
    return documents

"""## Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(documents):
    """Create FAISS vector store from documents"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

"""## Step 2 - Retrieval"""

def create_retriever(vector_store):
    """Create retriever from vector store"""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever

"""## Step 3 - Augmentation"""

def create_prompt():
    """Create the prompt template for video analysis"""
    prompt = PromptTemplate(
        template="""
You are a video transcript analysis assistant.

Answer STRICTLY using the transcript context below.
Do NOT use outside knowledge.

TASK:
1. Decide whether the user's topic is discussed in the video.
2. If YES:
   - Clearly explain what is discussed
   - Give a short summary relevant to the question
   - List the EXACT video timestamps where it is discussed
3. If NO:
   - Say only: "NO. Sorry, the topic is not discussed in this video."

RULES:
- Start your answer with YES or NO
- Timestamps MUST be in MM:SS format
- Use ONLY timestamps present in the context

Transcript context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )
    return prompt

"""## Step 4 - Generation"""

def format_docs(retrieved_docs):
    """Format retrieved documents into context string with timestamps"""
    def seconds_to_timestamp(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    context_text = "\n\n".join(
        f"[Timestamp: {seconds_to_timestamp(doc.metadata['start'])} "
        f"to {seconds_to_timestamp(doc.metadata['end'])}]\n"
        f"{doc.page_content}"
        for doc in retrieved_docs
    )
    return context_text

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain(retriever, prompt, chat_model):
    """Create the complete RAG chain"""
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | chat_model | parser
    return main_chain

def answer_question(main_chain, question):
    """Answer a question using the RAG chain"""
    answer = main_chain.invoke(question)
    return answer

if __name__ == "__main__":
    # Example usage
    video_id = "6S59Y0ckTm4"
    
    # Fetch transcript
    transcript = fetch_transcript(video_id)
    
    if transcript:
        # Create documents
        documents = create_documents_from_transcript(transcript)
        
        # Create vector store
        vector_store = create_vector_store(documents)
        
        # Create retriever
        retriever = create_retriever(vector_store)
        
        # Create prompt
        prompt = create_prompt()
        
        # Create RAG chain
        main_chain = create_rag_chain(retriever, prompt, chat_model)
        
        # Answer a question
        question = "is the topic about plastic discussed?"
        answer = answer_question(main_chain, question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
