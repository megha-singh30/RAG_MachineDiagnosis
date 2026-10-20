"""
Session 5 - Retrieval evaluation.
Measures recall@k and MRR of the retriever against a labelled gold set.

Run:  python evaluate.py        (uses chroma_db/ from Session 3 and gold_set.json)
"""

import json
from pathlib import Path

from langchain_community.vectorstores import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = "chroma_db"
GOLD = "gold_set.json"
K = 3


# reconnect to the vector store built in Session 3 (same embedding model).
def get_store():
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_DIR, embedding_function=emb)


# read the human-labelled gold set: list of {question, relevant_sources}.
def load_gold_set():
    return json.loads(Path(GOLD).read_text(encoding="utf-8"))


# score recall@k and MRR across the gold set.
def evaluate(store, gold, k=K):
    hits = 0                 # how many questions had a correct doc in top-k
    rr_total = 0.0           # sum of reciprocal ranks (for MRR)

    for item in gold:
        q = item["question"]
        relevant = set(item["relevant_sources"])
        results = store.similarity_search(q, k=k)
        got = [r.metadata.get("source", "?") for r in results]

        # rank (1-based) of the FIRST retrieved doc that is relevant
        rank = next((i + 1 for i, s in enumerate(got) if s in relevant), None)
        hit = rank is not None
        hits += hit
        rr_total += (1.0 / rank) if rank else 0.0

        print(f"[{'PASS' if hit else 'MISS'}] {q}")
        for i, s in enumerate(got, 1):
            mark = "*" if s in relevant else " "
            print(f"      {mark} #{i} {s}")

    n = len(gold)
    print(f"\nrecall@{k}: {hits}/{n} = {hits/n:.0%}")
    print(f"MRR:      {rr_total/n:.2f}")


if __name__ == "__main__":
    evaluate(get_store(), load_gold_set())
