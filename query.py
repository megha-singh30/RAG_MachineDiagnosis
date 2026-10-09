"""
Session 4 - Query pipeline (the online phase).
Question -> retrieve from ChromaDB -> grounded prompt -> cited answer.

Run:   python query.py "high 2x vibration with strong axial motion"
       (optional) set ANTHROPIC_API_KEY for an LLM-written answer; otherwise
       it runs in template mode (no key, no cost).
"""

import os
import sys

from langchain_community.vectorstores import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = "chroma_db"   # the folder ingest.py created in Session 3

GROUNDING = (
    "You are a rotating-equipment diagnostics assistant. Answer the question "
    "using ONLY the context below. If the context does not cover it, say so "
    "plainly. Cite the source of each claim by its [source]."
)


# FUNCTION 1 - reconnect to the vector store built in Session 3.
# The embedding model MUST match the one used at ingestion, or the query
# vector and the stored vectors live in different spaces and retrieval breaks.
def get_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


# FUNCTION 2 - retrieve the top-k most similar chunks to the question.
def retrieve(store, question, k=3):
    return store.similarity_search(question, k=k)


# FUNCTION 3 - AUGMENT: build the prompt = context + question + grounding rule.
def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[{c.metadata.get('source','?')}] {c.page_content}" for c in chunks
    )
    return f"{GROUNDING}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


# FUNCTION 4 - GENERATE: LLM if a key is set, else a deterministic template.
def generate(prompt, chunks):
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model="claude-haiku-4-5", max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    # template mode: assemble an answer straight from the retrieved chunks
    lines = ["(template mode - no LLM; set ANTHROPIC_API_KEY for phrased answers)\n"]
    for c in chunks:
        first = next(s for s in c.page_content.splitlines() if s.strip())
        lines.append(f"- [{c.metadata.get('source','?')}] {first.strip()}")
    return "\n".join(lines)


# FUNCTION 5 - orchestrate: retrieve -> prompt -> generate -> print with sources.
def answer_question(question, k=3):
    store = get_store()
    chunks = retrieve(store, question, k=k)
    print('/nChunks retrieved: ',chunks)
    prompt = build_prompt(question, chunks)
    print('/Prompt generated: ',prompt)
    answer = generate(prompt, chunks)
    print(f"Q: {question}\n")
    print(answer)
    print("\nSources:")
    for c in chunks:
        print(f"  - {c.metadata.get('source','?')} "
              f"[{c.metadata.get('doc_type','?')} | {c.metadata.get('fault','?')}]")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "high 2x vibration with strong axial motion"
    answer_question(q)