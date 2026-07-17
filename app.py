import streamlit as st
from google import genai
import chromadb
from chromadb.utils import embedding_functions

# --- 1. Streamlit UI Setup ---
st.set_page_config(page_title="Production-Ready RAG", layout="centered")
st.title("🚀 Production-Ready RAG System")
st.subheader("Learn how to build and query vector databases live!")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    # Let users paste some reference text/document live
    st.subheader("Feed Knowledge Base")
    raw_text = st.text_area(
        "Paste your document text here:", 
        placeholder="e.g., The official schedule for the RAG workshop..."
    )
    submit_button = st.button("Process & Index Document")

# Ensure API Key is provided
if not api_key:
    st.info("Please enter your Gemini API Key in the sidebar to start.", icon="🔑")
    st.stop()

# Initialize the modern GenAI Client
client = genai.Client(api_key=api_key)

# --- 2. Initialize ChromaDB (In-Memory) ---
@st.cache_resource
def get_vector_db():
    # In-memory database; resets every time the app restarts (perfect for quick demos!)
    client = chromadb.Client()
    # We use a default lightweight embedding model (all-MiniLM-L6-v2) run locally by Chroma
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(name="workshop_rag", embedding_function=emb_fn)

collection = get_vector_db()

# --- 3. Chunking & Ingestion ---
if submit_button and raw_text:
    with st.spinner("Chunking text and creating embeddings..."):
        # Basic paragraph-based chunking for demonstration
        chunks = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]
        
        # Ingest into ChromaDB
        if chunks:
            ids = [f"doc_chunk_{i}" for i in range(len(chunks))]
            collection.add(
                documents=chunks,
                ids=ids
            )
            st.sidebar.success(f"Successfully indexed {len(chunks)} text chunks!")
        else:
            st.sidebar.warning("No valid text chunks found to index.")

# --- 4. Chat Interface & Query Processing ---
st.divider()
user_query = st.text_input("Ask a question about your document:")

if user_query:
    with st.spinner("Searching knowledge base & generating answer..."):
        # Step A: Retrieve similar documents
        results = collection.query(
            query_texts=[user_query],
            n_results=2 # Retrieve top 2 most relevant chunks
        )
        
        retrieved_chunks = results.get('documents', [[]])[0]
        
        if retrieved_chunks:
            # Combine retrieved chunks to form context
            context = "\n---\n".join(retrieved_chunks)
            
            # Step B: Construct a secure, grounded system prompt
            prompt = f"""
            You are a helpful AI assistant. Answer the user's question using ONLY the provided context. 
            If the context does not contain the answer, politely state that you do not know.
            
            Context:
            {context}
            
            Question: {user_query}
            Answer:
            """
        
            
            # Call generate_content directly using the client and specify the model
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt 
                )


            # Step D: Display Results
            st.markdown("### 🤖 Answer")
            st.write(response.text)
            
            # Educational Segment: Show what was retrieved under the hood
            with st.expander("🔍 Behind the Scenes: See Retrieved Context Chunks"):
                for idx, chunk in enumerate(retrieved_chunks):
                    st.info(f"**Chunk {idx+1}:** {chunk}")
        else:
            st.warning("Knowledge base is empty! Paste and process some text in the sidebar first.")