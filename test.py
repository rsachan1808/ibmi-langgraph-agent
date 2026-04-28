# LangChain imports for RAG
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import RetrievalQA
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from dotenv import load_dotenv
from pathlib import Path
import os
import anthropic

# ── Load environment ──────────────────────────────────────
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
VOYAGE_KEY    = os.environ.get("VOYAGE_API_KEY")

# ── Build RAG pipeline ────────────────────────────────────
print("Loading IBMi documentation...")

loader    = TextLoader("RPG_test.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

embeddings  = VoyageAIEmbeddings(voyage_api_key=VOYAGE_KEY, model="voyage-3")
vectorstore = Chroma.from_documents(chunks, embeddings)

bm25_retriever   = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

rag_llm  = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    anthropic_api_key=ANTHROPIC_KEY
)
qa_chain = RetrievalQA.from_chain_type(
    llm=rag_llm,
    retriever=ensemble_retriever,
    return_source_documents=True
)

print("RAG pipeline ready")
