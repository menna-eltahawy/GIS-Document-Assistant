import streamlit as st
import os
import tempfile
import json
import time
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(page_title="GIS Document Assistant", page_icon="🗺️", layout="wide")

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "stats" not in st.session_state:
        st.session_state.stats = {"pdfs": 0, "chunks": 0, "questions": 0, "total_time": 0.0}

init_session_state()

st.title("🗺️ GIS Document Assistant")
st.markdown("RAG-powered chat with your documents — upload any PDF and start asking")

with st.sidebar:
    st.header("🔑 Authentication")
    
    provider = st.selectbox("Select Provider", ["Google Gemini", "Groq", "OpenRouter"])
    api_key = st.text_input(f"{provider} API Key", type="password")
    
    model_options = []
    if provider == "Google Gemini":
        model_options = ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
    elif provider == "Groq":
        model_options = ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
    elif provider == "OpenRouter":
        model_options = ["anthropic/claude-3-haiku", "meta-llama/llama-3-8b-instruct", "google/gemini-flash-1.5"]
        
    selected_model = st.selectbox("Select Model", model_options)
    
    st.divider()
    
    st.header("📄 Documents")
    uploaded_files = st.file_uploader("Drag and drop files here", type="pdf", accept_multiple_files=True)
    
    st.divider()
    
    st.header("📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PDFs", st.session_state.stats["pdfs"])
        st.metric("Questions", st.session_state.stats["questions"])
    with col2:
        st.metric("Chunks", st.session_state.stats["chunks"])
        avg_time = round(st.session_state.stats["total_time"] / max(1, st.session_state.stats["questions"]), 2)
        st.metric("Avg Response", f"{avg_time}s")
        
    st.divider()
    
    st.header("⚙️ Actions")
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with action_col2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.clear()
            st.rerun()

def get_llm():
    if not api_key:
        return None
    
    if provider == "Google Gemini":
        os.environ["GOOGLE_API_KEY"] = api_key
        return ChatGoogleGenerativeAI(model=selected_model, temperature=0.3)
    elif provider == "Groq":
        os.environ["GROQ_API_KEY"] = api_key
        return ChatGroq(model_name=selected_model, temperature=0.3)
    elif provider == "OpenRouter":
        os.environ["OPENROUTER_API_KEY"] = api_key
        return ChatOpenAI(
            model=selected_model, 
            temperature=0.3,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

if uploaded_files and st.session_state.vectorstore is None and api_key:
    with st.spinner("Processing documents..."):
        all_chunks = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            
            for chunk in chunks:
                chunk.metadata["source"] = uploaded_file.name
            all_chunks.extend(chunks)
            
        if all_chunks:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vectorstore = Chroma.from_documents(all_chunks, embeddings)
            
            st.session_state.stats["pdfs"] = len(uploaded_files)
            st.session_state.stats["chunks"] = len(all_chunks)
            st.rerun()

if not api_key:
    st.info("👈 Please enter your API Key in the sidebar to start.")
    st.stop()
elif not st.session_state.vectorstore:
    st.info("👈 Please upload at least one GIS PDF document to begin.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask a question about your documents...")

if len(st.session_state.messages) == 0:
    st.markdown("### Suggested Questions")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    if s_col1.button("What is this document about?", use_container_width=True):
        prompt = "What is this document about?"
    if s_col2.button("Summarize the key points", use_container_width=True):
        prompt = "Summarize the key points of the documents."
    if s_col3.button("List important terms", use_container_width=True):
        prompt = "List the most important technical terms mentioned."
    if s_col4.button("Give me a quick overview", use_container_width=True):
        prompt = "Give me a quick overview of the main topics."

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        start_time = time.time()
        with st.spinner("🧠 Analyzing documents..."):
            docs = st.session_state.vectorstore.similarity_search(prompt, k=4)
            context = "\n\n".join([d.page_content for d in docs])
            
            llm_prompt = f"""Use the following document context to answer the question accurately.
If the answer is not present in the context, explicitly state that you do not have enough information. Do not hallucinate.
CRITICAL INSTRUCTION: You must respond in the SAME LANGUAGE as the user's question.

Context:
{context}

Question: {prompt}

Answer:"""
            
            llm = get_llm()
            try:
                response = llm.invoke(llm_prompt)
                answer = response.content
                st.write(answer)
                
                with st.expander("Reference Sources"):
                    for i, doc in enumerate(docs, 1):
                        page = doc.metadata.get("page", "Unknown")
                        source_name = doc.metadata.get("source", "Unknown")
                        st.write(f"**Source {i}** (Page {page} from {source_name}):")
                        st.write(doc.page_content[:200] + "...")
                        
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                end_time = time.time()
                st.session_state.stats["questions"] += 1
                st.session_state.stats["total_time"] += (end_time - start_time)
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error communicating with the API. Please check your API key and try again.\n\nDetails: {str(e)}")