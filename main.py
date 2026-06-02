import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Literal
from typing_extensions import TypedDict
import yfinance as yf
import pandas as pd

# LangChain / LangGraph imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=".venv/.env")

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
os.environ["GROQ_API_KEY"]   = os.getenv("GROQ_API_KEY", "")
os.environ["HF_API_KEY"]     = os.getenv("HF_API_KEY", "")

st.set_page_config(page_title="Adaptive RAG - Financial Regulations", layout="wide")

# ── Custom Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0d1117; }
    [data-testid="stHeader"] { background: transparent; }
    h1 { color: #e6c87a !important; font-family: 'Georgia', serif; }
    h2, h3, h4 { color: #c9a84c !important; }
    .stTabs [data-baseweb="tab"] { color: #888; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #e6c87a !important; border-bottom: 2px solid #e6c87a; }
    .stButton > button { background: #1a2332; border: 1px solid #e6c87a; color: #e6c87a; }
    .stButton > button:hover { background: #e6c87a; color: #0d1117; }
    .stTextInput > div > div > input { background: #1a2332; color: #e0e0e0; border-color: #333; }
    .stDataFrame { background: #0d1117; }
    [data-testid="metric-container"] { background: #1a2332; border: 1px solid #2a3a4a;
        border-radius: 8px; padding: 12px; }
    [data-testid="metric-container"] label { color: #888 !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e6c87a !important; }
</style>
""", unsafe_allow_html=True)

st.title("🏦 Adaptive RAG: Financial & Banking Regulations Assistant")
st.subheader("Powered by LangGraph · FAISS · Groq · Tavily Search")

# ── Top 20 US Financial & Banking Companies ──────────────────────────────────
FINANCE_TICKERS = {
    "JPM":   "JPMorgan Chase",
    "BAC":   "Bank of America",
    "WFC":   "Wells Fargo",
    "GS":    "Goldman Sachs",
    "MS":    "Morgan Stanley",
    "C":     "Citigroup",
    "BLK":   "BlackRock",
    "SCHW":  "Charles Schwab",
    "AXP":   "American Express",
    "USB":   "U.S. Bancorp",
    "PNC":   "PNC Financial Services",
    "COF":   "Capital One Financial",
    "TFC":   "Truist Financial",
    "BK":    "Bank of New York Mellon",
    "STT":   "State Street Corp",
    "FI":    "Fiserv",
    "ICE":   "Intercontinental Exchange",
    "CME":   "CME Group",
    "SPGI":  "S&P Global",
    "MCO":   "Moody's Corp",
}


@st.cache_data(ttl=3600)
def fetch_finance_companies():
    """Fetch live financial data for top 20 US financial & banking companies."""
    data = []
    for ticker, name in FINANCE_TICKERS.items():
        try:
            info = yf.Ticker(ticker).info
            data.append({
                "Rank":       len(data) + 1,
                "Company":    name,
                "Ticker":     ticker,
                "Market Cap": info.get("marketCap", 0),
                "Revenue":    info.get("totalRevenue", 0),
                "Net Income": info.get("netIncomeToCommon", 0),
                "Employees":  info.get("fullTimeEmployees", "N/A"),
                "Sector":     info.get("sector", "Financial Services"),
                "52W High":   info.get("fiftyTwoWeekHigh", "N/A"),
                "52W Low":    info.get("fiftyTwoWeekLow", "N/A"),
                "P/E Ratio":  info.get("trailingPE", "N/A"),
                "Dividend %": info.get("dividendYield", "N/A"),
            })
        except Exception:
            continue

    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["Market Cap"] = df["Market Cap"].apply(lambda x: f"${x/1e9:.1f}B" if isinstance(x, (int, float)) and x else "N/A")
    df["Revenue"]    = df["Revenue"].apply(lambda x: f"${x/1e9:.1f}B" if isinstance(x, (int, float)) and x else "N/A")
    df["Net Income"] = df["Net Income"].apply(lambda x: f"${x/1e9:.1f}B" if isinstance(x, (int, float)) and x else "N/A")
    df["Employees"]  = df["Employees"].apply(lambda x: f"{x:,}" if isinstance(x, int) else x)
    df["P/E Ratio"]  = df["P/E Ratio"].apply(lambda x: f"{x:.1f}x" if isinstance(x, float) else x)
    df["Dividend %"] = df["Dividend %"].apply(lambda x: f"{x*100:.2f}%" if isinstance(x, float) else x)
    return df


@st.cache_resource
def get_embeddings():
    """Load HuggingFace embeddings once and reuse."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_retriever_from_upload(uploaded_file) -> object:
    """
    Save the uploaded PDF to a temp file, split it into chunks,
    build a FAISS vectorstore, and return the retriever.
    """
    import tempfile

    embd = get_embeddings()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader    = PyPDFLoader(tmp_path)
    pages     = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=50
    )
    doc_splits  = text_splitter.split_documents(pages)
    vectorstore = FAISS.from_documents(documents=doc_splits, embedding=embd)
    retriever   = vectorstore.as_retriever()

    os.unlink(tmp_path)   # clean up temp file
    return retriever, len(doc_splits)


@st.cache_resource
def initialize_rag_components(_retriever):
    """
    Builds and compiles the LangGraph workflow using the provided retriever.
    Call this after the user has uploaded a document and a retriever is ready.
    """

    if not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        st.warning("⚠️ Environment keys missing. Please check your .env file.")

    # ── Tools & Models ────────────────────────────────────────────────────────
    web_search_tool = TavilySearchResults(k=3)

    class RouteQuery(BaseModel):
        datasource: Literal["vectorstore", "web_search"] = Field(
            ..., description="Route to web search or vectorstore."
        )

    class GradeDocuments(BaseModel):
        binary_score: str = Field(description="Document is relevant to the question, 'yes' or 'no'")

    class GradeHallucinations(BaseModel):
        binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

    class GradeAnswer(BaseModel):
        binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.environ["GROQ_API_KEY"])

    structured_llm_router        = llm.with_structured_output(RouteQuery)
    structured_llm_grader        = llm.with_structured_output(GradeDocuments)
    structured_llm_hallucination = llm.with_structured_output(GradeHallucinations)
    structured_llm_answer        = llm.with_structured_output(GradeAnswer)

    # Prompts — finance-domain tuned
    question_router = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents about financial regulations, banking supervision, model risk management,
Basel III, Dodd-Frank, Fed guidance, and OCC/FDIC rules.
Use the vectorstore for questions on these topics. Otherwise use web-search."""),
        ("human", "{question}")
    ]) | structured_llm_router

    retrieval_grader = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing relevance of a retrieved document to a user question
about financial regulations or banking. Grade 'yes' if the document is relevant, 'no' otherwise."""),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}")
    ]) | structured_llm_grader

    rag_chain = ChatPromptTemplate.from_messages([
        ("system", "You are a financial regulatory expert assistant. "
                   "Use the retrieved context to answer concisely in three sentences max. "
                   "If you don't know, say so."),
        ("human", "Question: {question}\nContext: {context}\nAnswer:"),
    ]) | llm | StrOutputParser()

    hallucination_grader = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing whether an LLM generation is grounded in retrieved facts. "
                   "Give a binary score 'yes' or 'no'."),
        ("human", "Set of facts:\n\n{documents}\n\nLLM generation: {generation}")
    ]) | structured_llm_hallucination

    answer_grader = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing whether an answer resolves a question. "
                   "Give a binary score 'yes' or 'no'."),
        ("human", "User question:\n\n{question}\n\nLLM generation: {generation}")
    ]) | structured_llm_answer

    question_rewriter = ChatPromptTemplate.from_messages([
        ("system", "You are a question re-writer that improves questions for financial regulation "
                   "vectorstore retrieval."),
        ("human", "Initial question:\n\n{question}\n\nFormulate an improved question.")
    ]) | llm | StrOutputParser()

    # C. Graph Nodes
    def retrieve(state):
        documents = _retriever.invoke(state["question"])
        return {"documents": documents, "question": state["question"]}

    def generate(state):
        generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
        return {"documents": state["documents"], "question": state["question"], "generation": generation}

    def grade_documents(state):
        filtered = []
        for d in state["documents"]:
            score = retrieval_grader.invoke({"question": state["question"], "document": d.page_content})
            if score.binary_score == "yes":
                filtered.append(d)
        return {"documents": filtered, "question": state["question"]}

    def transform_query(state):
        better_question = question_rewriter.invoke({"question": state["question"]})
        return {"documents": state["documents"], "question": better_question}

    def web_search(state):
        docs        = web_search_tool.invoke({"query": state["question"]})
        web_results = Document(page_content="\n".join([d["content"] for d in docs]))
        return {"documents": [web_results], "question": state["question"]}

    # Conditional edges
    def route_question(state):
        source = question_router.invoke({"question": state["question"]})
        return "web_search" if source.datasource == "web_search" else "vectorstore"

    def decide_to_generate(state):
        return "transform_query" if not state["documents"] else "generate"

    def grade_generation_v_documents_and_question(state):
        h_score = hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]})
        if h_score.binary_score == "yes":
            a_score = answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
            return "useful" if a_score.binary_score == "yes" else "not useful"
        return "not supported"

    # D. Build Graph
    class GraphState(TypedDict):
        question:   str
        generation: str
        documents:  List[Document]

    workflow = StateGraph(GraphState)
    workflow.add_node("web_search",      web_search)
    workflow.add_node("retrieve",        retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate",        generate)
    workflow.add_node("transform_query", transform_query)

    workflow.add_conditional_edges(START, route_question,
        {"web_search": "web_search", "vectorstore": "retrieve"})
    workflow.add_edge("web_search",      "generate")
    workflow.add_edge("retrieve",        "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"})
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges("generate", grade_generation_v_documents_and_question,
        {"not supported": "generate", "useful": END, "not useful": "transform_query"})

    return workflow.compile()


# ── UI ───────────────────────────────────────────────────────────────────────
st.markdown("---")
tab1, tab2 = st.tabs(["💬 Financial Regulations Assistant", "🏦 Top 20 Financial & Banking Companies 2026"])

# ── Tab 1: Upload → RAG Assistant ─────────────────────────────────────────────
with tab1:

    # ── Step 1: Document Upload ───────────────────────────────────────────────
    st.markdown("### 📁 Step 1 — Upload your Financial Regulation PDF")
    st.caption("Upload any financial regulation document (Basel III, Dodd-Frank, Fed guidance, OCC rules, etc.)")

    uploaded_file = st.file_uploader(
        "Drop a PDF here or click to browse",
        type=["pdf"],
        key="pdf_uploader"
    )

    if uploaded_file is not None:
        # Only re-process if a new file was uploaded
        if st.session_state.get("last_uploaded_filename") != uploaded_file.name:
            with st.spinner(f"📄 Processing **{uploaded_file.name}** — splitting & indexing into FAISS..."):
                try:
                    retriever, num_chunks = build_retriever_from_upload(uploaded_file)
                    st.session_state["retriever"]              = retriever
                    st.session_state["last_uploaded_filename"] = uploaded_file.name
                    st.session_state["num_chunks"]             = num_chunks
                    # Clear cached graph so it rebuilds with the new retriever
                    initialize_rag_components.clear()
                    st.session_state["app_engine"] = initialize_rag_components(retriever)
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")

        if st.session_state.get("last_uploaded_filename") == uploaded_file.name:
            st.success(
                f"✅ **{uploaded_file.name}** indexed — "
                f"{st.session_state.get('num_chunks', '?')} chunks loaded into FAISS vectorstore."
            )

    # ── Step 2: Ask Questions ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💬 Step 2 — Ask a question about the uploaded document")

    if "app_engine" not in st.session_state:
        st.info("⬆️ Please upload a PDF above to activate the agent.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 What is model risk management?"):
                st.session_state.user_query = "What is model risk management in banking?"
        with col2:
            if st.button("🏛️ What are Basel III capital requirements?"):
                st.session_state.user_query = "What are Basel III capital requirements for banks?"
        with col3:
            if st.button("⚖️ What is the Dodd-Frank Act?"):
                st.session_state.user_query = "What is the Dodd-Frank Wall Street Reform Act?"

        user_input = st.text_input(
            "Enter your question here:",
            value=st.session_state.get("user_query", ""),
            placeholder="e.g. What are the key components of model validation?",
            key="tab1_input"
        )

        if st.button("🚀 Run Adaptive Agent", type="primary", key="tab1_run"):
            if user_input.strip() == "":
                st.error("Please enter a valid query.")
            else:
                if "user_query" in st.session_state:
                    del st.session_state.user_query

                with st.spinner("Agent reasoning · retrieving · self-correcting via Adaptive RAG..."):
                    try:
                        output  = st.session_state["app_engine"].invoke({"question": user_input})
                        res_col, doc_col = st.columns([3, 2])

                        with res_col:
                            st.success("### ✅ Final Generation")
                            st.write(output.get("generation", "No generation produced."))
                            st.info(f"**Processed Question:** {output.get('question')}")

                        with doc_col:
                            st.markdown("### 📂 Contextual Sources Used")
                            retrieved_docs = output.get("documents", [])
                            if isinstance(retrieved_docs, list):
                                for idx, doc in enumerate(retrieved_docs):
                                    meta      = getattr(doc, "metadata", {})
                                    page_info = f" | Page {meta.get('page_label', idx)}" if meta.get("page_label") else ""
                                    with st.expander(f"Source [{idx+1}]{page_info}"):
                                        st.caption(f"Source: {meta.get('source', uploaded_file.name)}")
                                        st.write(doc.page_content)
                            else:
                                with st.expander("Web Context Data"):
                                    st.write(str(retrieved_docs))
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

# ── Tab 2: Top 20 Financial Companies ────────────────────────────────────────
with tab2:
    st.markdown("#### 📊 Top 20 US Financial & Banking Companies — Live Data")
    st.caption("Data sourced from Yahoo Finance via yfinance · Refreshes every hour")

    if st.button("🔄 Fetch / Refresh Data", type="primary", key="tab2_refresh"):
        st.cache_data.clear()

    with st.spinner("Fetching live data from Yahoo Finance..."):
        df = fetch_finance_companies()

    # Summary metrics
    st.markdown("### 📈 Market Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Companies Tracked", len(df))
    m2.metric("Largest by Market Cap", df.iloc[0]["Company"] if not df.empty else "N/A")
    m3.metric("Sectors Covered", "Banks · Asset Mgmt · Exchanges · Ratings")
    m4.metric("Data Source", "Yahoo Finance")

    st.markdown("---")

    # Sector breakdown
    st.markdown("### 🏦 Company Rankings & Financials")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download
    csv = df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv,
        file_name="top20_financial_banking_companies_2026.csv",
        mime="text/csv",
        key="tab2_download"
    )

    st.markdown("---")
    st.markdown("### 📘 About the Companies")
    st.markdown("""
| Category | Companies |
|---|---|
| **Money Center Banks** | JPMorgan Chase, Bank of America, Wells Fargo, Citigroup |
| **Investment Banks** | Goldman Sachs, Morgan Stanley |
| **Asset Management** | BlackRock, State Street |
| **Retail / Consumer Finance** | Charles Schwab, American Express, Capital One |
| **Regional Banks** | U.S. Bancorp, PNC Financial, Truist Financial |
| **Custody Banks** | Bank of New York Mellon |
| **Financial Technology** | Fiserv |
| **Exchanges & Data** | Intercontinental Exchange, CME Group, S&P Global, Moody's |
""")