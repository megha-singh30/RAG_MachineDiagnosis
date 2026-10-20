"""
Session 5 - Retrieval evaluation with difficulty breakdown.
Measures recall@k and MRR of the retriever against a labelled gold set,
reported overall AND split by difficulty (easy vs hard/paraphrased).

Run:  python evaluate.py
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


def get_store():
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_DIR, embedding_function=emb)


def load_gold_set():
    return json.loads(Path(GOLD).read_text(encoding="utf-8"))


def _score_one(store, item, k):
    """Return (hit, reciprocal_rank, retrieved_sources) for one question."""
    relevant = set(item["relevant_sources"])
    got = [r.metadata.get("source", "?") for r in store.similarity_search(item["question"], k=k)]
    rank = next((i + 1 for i, s in enumerate(got) if s in relevant), None)
    return (rank is not None), (1.0 / rank if rank else 0.0), got


def evaluate(store, gold, k=K):
    # buckets: 'all' plus one per difficulty label present in the gold set
    buckets = {}
    for item in gold:
        hit, rr, got = _score_one(store, item, k)
        for key in ("all", item.get("difficulty", "unlabelled")):
            b = buckets.setdefault(key, {"hits": 0, "rr": 0.0, "n": 0})
            b["hits"] += hit; b["rr"] += rr; b["n"] += 1

        tag = item.get("difficulty", "?")
        print(f"[{'PASS' if hit else 'MISS'}] ({tag}) {item['question']}")
        for i, s in enumerate(got, 1):
            mark = "*" if s in set(item["relevant_sources"]) else " "
            print(f"      {mark} #{i} {s}")

    print("\n--- Results ---")
    for key in ["all", "easy", "hard"]:
        if key in buckets:
            b = buckets[key]
            print(f"{key:5}  recall@{k}: {b['hits']}/{b['n']} = {b['hits']/b['n']:.0%}"
                  f"   MRR: {b['rr']/b['n']:.2f}")


if __name__ == "__main__":
    evaluate(get_store(), load_gold_set())