"""
Session 3 - Ingestion pipeline.
Turns the corpus into a searchable ChromaDB vector store.

Flow:  load_documents() -> chunk_documents() -> build_vector_store()
Run:   python ingest.py      (creates ./chroma_db, then runs a test query)
"""

from pathlib import Path
import pandas as pd
from langchain_chroma import Chroma

# LangChain building blocks
from langchain_core.documents import Document          # a Document = text + metadata
from langchain_community.vectorstores import Chroma    # the vector database

# The text splitter moved packages across LangChain versions -> try both.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# The embedding model wrapper also moved -> try the new package, fall back.
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

DATA = Path("data")     # folder holding guidance/, logs/, fault_table.csv
DB_DIR = "chroma_db"    # where the vector database is written on disk


# --------------------------------------------------------------------------
# FUNCTION 1 - load every file into a uniform list of Document objects.
# A Document has just two parts: page_content (the text) and metadata (a dict).
#
# --------------------------------------------------------------------------
def load_documents():
    docs = []

    # Guidance files (UNSTRUCTURED). One fault per file -> filename = fault name.
    for path in sorted((DATA / "guidance").glob("*.txt")):
        docs.append(Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.name, "doc_type": "guidance", "fault": path.stem},
        ))

    # Maintenance logs (SEMI-STRUCTURED). Fault not tagged yet -> "unknown".
    for path in sorted((DATA / "logs").glob("*.txt")):
        docs.append(Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.name, "doc_type": "log", "fault": "unknown"},
        ))

    # Fault table (STRUCTURED). ONE Document per ROW so each fault is retrievable.
    df = pd.read_csv(DATA / "fault_table.csv")
    for _, row in df.iterrows():
        # turn a row into a readable sentence: "fault: Unbalance; primary_frequency: 1x; ..."
        text = "; ".join(f"{col}: {row[col]}" for col in df.columns)
        docs.append(Document(
            page_content=text,
            metadata={"source": "fault_table.csv", "doc_type": "fault_table", "fault": row["fault"]},
        ))

    return docs


# --------------------------------------------------------------------------
# FUNCTION 2 - split Documents into chunks.
# chunk_size=1000 chars keeps each of our short files WHOLE (one topic per
# chunk); chunk_overlap=100 would carry context across a split IF one happened.
# split_documents keeps each chunk's metadata attached automatically.
# --------------------------------------------------------------------------
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)


# --------------------------------------------------------------------------
# FUNCTION 3 - embed every chunk and store it in ChromaDB on disk.
# HuggingFaceEmbeddings wraps the same all-MiniLM-L6-v2 model from Session 1.
# Chroma.from_documents: embeds each chunk's text, stores vector+text+metadata,
# and persists to DB_DIR so you build once and reuse.
# --------------------------------------------------------------------------
def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    return store


# --------------------------------------------------------------------------
# FUNCTION 4 - run the pipeline and prove it works with a test query.
# --------------------------------------------------------------------------
def main():
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    store = build_vector_store(chunks)
    print(f"Stored in ChromaDB at ./{DB_DIR}")

    query = "high 2x vibration with strong axial motion"
    print(f"\nTest query: '{query}'")
    for i, d in enumerate(store.similarity_search(query, k=3), 1):
        meta = d.metadata
        print(f"  #{i} [{meta['doc_type']} | {meta['fault']}] {d.page_content[:60].strip()}...")


if __name__ == "__main__":
    main()
