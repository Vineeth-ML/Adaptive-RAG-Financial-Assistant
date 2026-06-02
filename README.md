🏦 Adaptive RAG for Financial Regulations
Overview
This project is an Adaptive Retrieval-Augmented Generation (RAG) application built with Streamlit, LangGraph, FAISS, Groq, and Tavily Search.
Users can upload financial regulation PDFs (Basel III, Dodd-Frank, OCC, FDIC, Fed guidance, etc.) and ask questions in natural language. The system retrieves relevant information from the document and generates accurate answers using an LLM.
The application also provides live financial data for the top U.S. banking and financial institutions.
Features
📄 Upload and analyze PDF documents
🔍 Semantic search using FAISS vector database
🤖 Adaptive RAG workflow with LangGraph
🌐 Automatic web search fallback using Tavily
✅ Document relevance grading
✅ Hallucination detection and answer validation
📊 Live financial market data using Yahoo Finance
🎨 Interactive Streamlit dashboard
Tech Stack
Streamlit
LangChain
LangGraph
FAISS
Groq (Llama 3.3 70B)
Hugging Face Embeddings
Tavily Search
Yahoo Finance (yfinance)
Pandas
Installation
Clone Repository
git clone <your-repository-url>
cd adaptive-rag-finance
Create Virtual Environment
uv venv
source .venv/bin/activate
Install Dependencies
uv pip install -r requirements.txt
Environment Variables
Create a .env file inside your project directory:
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
HF_API_KEY=your_huggingface_api_key
Run the Application
streamlit run app.py
Open:
http://localhost:8501
How It Works
Upload a financial regulation PDF.
The document is split into chunks.
Embeddings are created using Hugging Face.
Chunks are stored in a FAISS vector database.
User asks a question.
LangGraph routes the query to:
Vector Search (document knowledge)
Web Search (Tavily) if needed
The answer is generated and validated before being displayed.
Example Questions
What is model risk management?
What are Basel III capital requirements?
What is the Dodd-Frank Act?
What are OCC model validation requirements?
Explain liquidity coverage ratio (LCR).
Future Improvements
Multi-document support
SEC filing analysis
Financial news summarization
Hybrid search (BM25 + Vector Search)
Compliance monitoring assistant
Author
Vineeth Raju
AI-powered Financial Regulations Assistant using Adaptive RAG, LangGraph, FAISS, and Groq LLMs.