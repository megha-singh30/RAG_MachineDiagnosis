"""Session 1 - Hello World RAG. One document, one query, one grounded answer."""

import os
import numpy as np

# --- 1. ONE document ---
DOCUMENT = """
Elevated vibration at exactly 1x running speed usually indicates rotor mass
unbalance; the fix is to balance the rotor. When the 1x component is
accompanied by strong harmonics (2x, 3x) and high axial vibration, suspect
shaft misalignment and check the coupling alignment. A rising 1x amplitude
that grows slowly over weeks often points to a developing unbalance from
fouling or wear. Bearing defects show up not at 1x but at high-frequency
bearing defect frequencies (BPFO, BPFI) and as raised broadband noise.
""".strip()

# --- 2. Split the document into chunks (sentences) ---
def split_into_chunks(text):
    return [s.strip() + "." for s in text.replace("\n", " ").split(".") if s.strip()]

# --- 3. Embeddings: real stack if available, else local fallback ---
def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return "sentence-transformers", lambda xs: np.array(model.encode(list(xs)))
    except Exception:
        from sklearn.feature_extraction.text import HashingVectorizer
        v = HashingVectorizer(n_features=256, stop_words="english",
                              alternate_sign=False, norm="l2")
        return "local-hashing (fallback)", lambda xs: v.transform(list(xs)).toarray()

def cosine(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return a @ b

# --- 4. The RAG loop ---
def main():
    query = "high vibration at 1x running speed with strong harmonics"
    chunks = split_into_chunks(DOCUMENT)
    backend, embed = get_embedder()
    chunk_vecs = embed(chunks)
    query_vec = embed([query])[0]

    # RETRIEVE
    scores = cosine(chunk_vecs, query_vec)
    ranking = np.argsort(scores)[::-1]          # best to worst
    print("Top 3 retrieved chunks:")
    for rank, idx in enumerate(ranking[:3], 1):
        print(f"  #{rank}  score={scores[idx]:.3f}  {chunks[idx]}")
    retrieved = chunks[int(ranking[0])]

    # AUGMENT
    prompt = ("Answer the question using ONLY the context. If the context does "
              "not cover it, say so.\n\n"
              f"Context:\n{retrieved}\n\nQuestion: {query}\nAnswer:")

    print(f"Embedding backend: {backend}")
    print(f"Query: {query}\n")
    print("Retrieved chunk (highest similarity):")
    print(f"  score={scores[ranking[0]]:.3f}  {retrieved}\n")

    # GENERATE
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model="claude-haiku-4-5", max_tokens=200,
            messages=[{"role": "user", "content": prompt}])
        print("Generated answer:\n", msg.content[0].text)
    else:
        print("Generated answer (no API key -> showing grounded context):")
        print(" ", retrieved)
        print("\n[Set ANTHROPIC_API_KEY to have an LLM phrase this for you.]")

if __name__ == "__main__":
    main()