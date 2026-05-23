# 🗺️ GIS Document Assistant

A modern, professional RAG-powered Streamlit web application designed to chat with GIS PDF documents using multiple LLM providers (Google Gemini, Groq, OpenRouter).

## 📋 Core Features

| Feature | Description |
| :--- | :--- |
| **1. Upload PDF** | Upload GIS-related PDFs (Standards, Tutorials, Manuals). |
| **2. Process** | Divide documents into chunks and store them in a vector database. |
| **3. Chat** | Ask questions and get accurate answers based on the document context. |
| **4. Sources** | Display the exact sources and pages referenced for each answer. |

## 🏆 Bonus Features

| Status | Feature |
| :---: | :--- |
| ✅ | **Multi-PDF Support** (Ability to upload and process multiple files simultaneously). |
| ✅ | **Export Chat to JSON** (Download the conversation history as a JSON file). |
| ✅ | **Bilingual Support (English/Arabic)** (Dual-language UI with a toggle switch). |
| ✅ | **Session Statistics** (Track questions, files, average response time, and chunks). |

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
