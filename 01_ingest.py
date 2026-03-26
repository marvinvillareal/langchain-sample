"""
Step 1: Document Ingestion
--------------------------
Loads documents from ./docs, chunks them, embeds them with Ollama,
and persists them to a local Chroma vector store.

Usage:
    mkdir docs
    # Add your .txt files to ./docs
    python 01_ingest.py
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def ingest():
    print("📂 Loading documents...")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    raw_docs = loader.load()
    print(f"   Loaded {len(raw_docs)} document(s)")

    print("\n✂️  Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Tries each separator in order; falls back to the next one
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.split_documents(raw_docs)
    print(f"   Created {len(docs)} chunk(s)")

    print("\n🔢 Embedding and storing in Chroma...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"   ✅ Persisted vector store to '{CHROMA_DIR}'")
    return vectorstore

def get_vector_data(vectorstore: Chroma):
    print("\n📊 Retrieving vector data for verification...")
    try:
        collection = vectorstore.get()
        print(f"   Collection name: {collection}")
        print(f"   Number of vectors: {collection['ids']}")
        # all_data = collection.get()
        # print(f"data: {all_data}")

    except Exception as e:
        print(f"   Error retrieving collection: {e}")
    



if __name__ == "__main__":
    vectorstore = ingest()
    get_vector_data(vectorstore)
