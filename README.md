# 🏦 Adaptive RAG for Financial Regulations

## Overview

This project is an **Adaptive Retrieval-Augmented Generation (RAG)** application built with **Streamlit, LangGraph, FAISS, Groq, and Tavily Search**.

Users can upload financial regulation PDFs (Basel III, Dodd-Frank, OCC, FDIC, Fed guidance, etc.) and ask questions in natural language. The system retrieves relevant information from the document and generates accurate answers using a large language model (LLM).

The application also provides **live financial data** for top U.S. banking and financial institutions.

---

## Features

- 📄 Upload and analyze PDF documents  
- 🔍 Semantic search using FAISS vector database  
- 🤖 Adaptive RAG workflow with LangGraph  
- 🌐 Automatic web search fallback using Tavily  
- ✅ Document relevance grading  
- ✅ Hallucination detection and answer validation  
- 📊 Live financial market data using Yahoo Finance  
- 🎨 Interactive Streamlit dashboard  

---

## Tech Stack

- Streamlit  
- LangChain  
- LangGraph  
- FAISS  
- Groq (Llama 3.3 70B)  
- Hugging Face Embeddings  
- Tavily Search API  
- Yahoo Finance (yfinance)  
- Pandas  

---

## Installation

### Clone Repository

```bash
git clone <your-repository-url>
cd adaptive-rag-finance
