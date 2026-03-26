"""
Step 2: Retrieval Strategies
-----------------------------
Demonstrates four retrieval approaches against the Chroma vector store
built in step 1. Run 01_ingest.py first.

Usage:
    python 02_retrieval.py
"""

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
TEST_QUERY = "What is authentication?"


def load_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def demo_basic(vectorstore: Chroma):
    print("\n── 1. Basic Similarity Search ──────────────────────────────────")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    docs = retriever.invoke(TEST_QUERY)
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"  [{i}] ({source})\n      {doc.page_content[:120]}...")


def demo_mmr(vectorstore: Chroma):
    print("\n── 2. MMR – Maximal Marginal Relevance (reduces redundancy) ────")
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 15,   # candidates to consider before re-ranking
            "lambda_mult": 0.5,  # 0 = max diversity, 1 = max relevance
        },
    )
    docs = retriever.invoke(TEST_QUERY)
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"  [{i}] ({source})\n      {doc.page_content[:120]}...")


def demo_multi_query(vectorstore: Chroma):
    print("\n── 3. Multi-Query Retriever (auto-rephrases the question) ──────")
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm,
    )
    # Enable logging to see the generated queries
    import logging
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    docs = retriever.invoke(TEST_QUERY)
    print(f"  Retrieved {len(docs)} unique chunk(s) across all rephrased queries")
    for i, doc in enumerate(docs[:3], 1):
        print(f"  [{i}] {doc.page_content[:120]}...")


def demo_compression(vectorstore: Chroma):
    print("\n── 4. Contextual Compression (extracts only relevant passages) ─")
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    )
    docs = retriever.invoke(TEST_QUERY)
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.page_content[:200]}...")


if __name__ == "__main__":
    print(f"🔍 Query: '{TEST_QUERY}'")
    vs = load_vectorstore()
    demo_basic(vs)
    demo_mmr(vs)
    demo_multi_query(vs)
    demo_compression(vs)
