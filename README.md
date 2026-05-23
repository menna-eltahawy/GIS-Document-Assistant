# GIS Document Assistant

A modern, professional RAG-powered Streamlit web application designed to chat with GIS PDF documents using multiple LLM providers (Google Gemini, Groq, OpenRouter).

## Features
- **Multi-LLM Support**: Seamless switching between Google Gemini, Groq, and OpenRouter with dynamic model selection.
- **Local Vector Embeddings**: Uses HuggingFace's `all-MiniLM-L6-v2` via `sentence-transformers` for local embedding calculation without relying on external API credits.
- **Interactive Dashboard**: Clean, dark-themed UI featuring multi-file uploading, metrics tracking, and quick suggestion chips.
- **Session Tracking**: Real-time evaluation metrics displaying total PDFs, generated chunks, queries asked, and average response times.
- **Quick Controls**: One-click actions to clear active chat sessions or completely reset runtime storage.

## Project Structure
- `gis_rag_app.py`: The main Streamlit web application.
- `requirements.txt`: Python package dependencies list.
- `.env`: Environment variables configuration for API keys.
- `.gitignore`: Specifies intentionally untracked files to ignore.

## Installation & Setup

1. **Clone or Extract the Project Files** into a clean directory.
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Execute the following command to spin up the local development server:
```bash
streamlit run gis_rag_app.py
```
