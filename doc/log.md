# *Session -1 *

## DAY-1 --> 04-08-2026
1. Setting up the environment
2. Installing all packages
3. Running the file hello_rag.py

### Hello_rag.py
it contains all steps
chunking
embedding
vectorize

retrieval with new prompt

## Day-2 --> 05-08-2026

Understanding:  
(a) the two-phase model — ingestion is offline/once, retrieval+generation is online/per-query;  
(b) chunking ≠ embedding — split first, vectorize second;  
(c) your corpus has three doc types and why you mixed them.

Here are the three Session 2 quiz questions with clean, complete answers to paste into your log.

## Q1. Why put each fault in its own file instead of one big "all-faults" document?  
Because chunking splits text at fixed sizes, not at topic boundaries. In a single large document, a chunk can straddle the seam between two faults — the end of "unbalance" and the start of "misalignment" — producing one vector that represents two faults muddled together, which pollutes retrieval. One fault per file guarantees every chunk covers exactly one topic. It also makes metadata trivial: one file = one fault label to tag its chunks with.

## Q2. What are the three document types in the corpus, and which is the structured data?  
Unstructured (the guidance .txt files, written as prose), semi-structured (the maintenance logs — consistent fields like date/asset/symptom/action, but free text inside), and structured (the fault_table.csv — rows and columns). The CSV is the structured data. Mixing all three is the "integrate structured and unstructured data" capability that target roles ask for.

## Q3. If a user asks about a fault whose answer isn't in the corpus, what happens — and what does that say about corpus quality?  
The retriever never returns nothing — it's a nearest-neighbor search, so it always returns the top-k closest chunks, even when the true answer isn't present. For an out-of-corpus question those chunks are irrelevant-but-nearest. Whether the system correctly says "not in the provided context" depends entirely on the grounding instruction and the model honoring it; a weak prompt or model will instead answer from the wrong chunks — a hallucination. The lesson: answer quality is capped by corpus coverage. Gaps become refusals in the best case and hallucinations in the worst