# RAG repo skeleton — ready-to-run

This canvas now includes the repository skeleton with **five working Python scripts**: `ingest.py`, `chunker.py`, `embed_index.py`, `retrieve.py`, and `generate.py`. It also has `requirements.txt` and `README.md`.

---

## File: README.md

```markdown
# RAG Project — Repo Skeleton

This repository contains a minimal Retrieval-Augmented Generation (RAG) pipeline:

Files:
- `data/` - put your raw PDFs / text files here
- `src/ingest.py` - PDF/text ingestion and basic cleaning
- `src/chunker.py` - token-aware chunker
- `src/embed_index.py` - builds sentence-transformers embeddings + FAISS index
- `src/retrieve.py` - query interface to search FAISS index
- `src/generate.py` - simple RAG pipeline: retrieve + construct prompt + generate with local LLM
- `requirements.txt` - Python dependencies

Quick start:
1. Create a Python 3.10+ venv and install requirements: `pip install -r requirements.txt`
2. Put your PDF/text files inside `data/raw/`.
3. Run ingestion: `python src/ingest.py --input_dir data/raw --out_dir data/parsed`
4. Chunk the parsed texts: `python src/chunker.py --input_dir data/parsed --out_dir data/chunks`
5. Build the FAISS index: `python src/embed_index.py --chunks_dir data/chunks --index_path data/index.faiss --embeddings_path data/embeddings.npy`
6. Run retrieval: `python src/retrieve.py --index_path data/index.faiss --id_map data/chunks/id_map.txt --query "Your question here"`
7. Run RAG generation: `python src/generate.py --index_path data/index.faiss --id_map data/chunks/id_map.txt --query "Your question here"`
```

---

## File: requirements.txt

```
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
transformers>=4.30.0
torch>=2.0.0
tqdm
nltk
pdfplumber
python-multipart
pandas
scikit-learn
rank_bm25
```

---

## File: src/ingest.py

```python
# (same as before)
```

---

## File: src/chunker.py

```python
# (same as before)
```

---

## File: src/embed\_index.py

```python
# (same as before)
```

---

## File: src/retrieve.py

```python
"""
Retrieve top-k chunks from a FAISS index given a query string.
Usage:
  python src/retrieve.py --index_path data/index.faiss --id_map data/chunks/id_map.txt --query "What is ...?" --k 5
"""
import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_id_map(id_map_path):
    with open(id_map_path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--id_map", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    query_emb = model.encode([args.query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    index = faiss.read_index(args.index_path)
    D, I = index.search(query_emb, args.k)

    ids = load_id_map(args.id_map)
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        print(f"Rank {rank+1}: {ids[idx]} (score={score:.4f})")

if __name__ == "__main__":
    main()
```

---

## File: src/generate.py

```python
"""
Minimal RAG: retrieve top-k chunks and generate answer using a local model.
Usage:
  python src/generate.py --index_path data/index.faiss --id_map data/chunks/id_map.txt --query "What is ...?" --k 5
"""
import argparse
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def load_id_map(id_map_path):
    with open(id_map_path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--id_map", required=True)
    parser.add_argument("--chunks_dir", default="data/chunks")
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--gen_model", default="distilgpt2")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    # load retrieval index
    embed_model = SentenceTransformer(args.model)
    query_emb = embed_model.encode([args.query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    index = faiss.read_index(args.index_path)
    D, I = index.search(query_emb, args.k)

    ids = load_id_map(args.id_map)
    retrieved = []
    for idx in I[0]:
        cid = ids[idx]
        chunk_file = Path(args.chunks_dir) / f"{cid}.json"
        if chunk_file.exists():
            obj = json.loads(chunk_file.read_text(encoding="utf-8"))
            retrieved.append(obj)

    context = "\n\n".join([f"[{r['chunk_id']}] {r['text']}" for r in retrieved])
    prompt = f"Answer the question using only the context. Cite chunk IDs.\n\nContext:\n{context}\n\nQuestion: {args.query}\nAnswer:"

    generator = pipeline("text-generation", model=args.gen_model)
    output = generator(prompt, max_length=300, do_sample=False)[0]["generated_text"]
    print("--- Prompt ---\n", prompt)
    print("--- Answer ---\n", output)

if __name__ == "__main__":
    main()
```

---

## Notes & next steps

* `retrieve.py` returns ranked chunk IDs and scores.
* `generate.py` builds a naive RAG prompt and uses a local Hugging Face model (default: `distilgpt2`). Replace `--gen_model` with a stronger local LLM if you have GPU.
* Next enhancements: add hybrid retriever (BM25 + dense), add hallucination checker, and containerize with Docker.
