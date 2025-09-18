# RAG repo skeleton — ready-to-run

This canvas contains a ready-to-use repository skeleton and three working Python scripts: `ingest.py`, `chunker.py`, and `embed_index.py`. It also includes `requirements.txt` and `README.md` with instructions to run locally.

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
- `requirements.txt` - Python dependencies

Quick start:
1. Create a Python 3.10+ venv and install requirements: `pip install -r requirements.txt`
2. Put your PDF/text files inside `data/raw/`.
3. Run ingestion: `python src/ingest.py --input_dir data/raw --out_dir data/parsed`
4. Chunk the parsed texts: `python src/chunker.py --input_dir data/parsed --out_dir data/chunks`
5. Build the FAISS index: `python src/embed_index.py --chunks_dir data/chunks --index_path data/index.faiss`

See each script's help for more options.
```

---

## File: requirements.txt

```
# minimal requirements
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
"""
Ingest raw PDF and TXT files from a directory and write cleaned plain-text files
Usage:
  python src/ingest.py --input_dir data/raw --out_dir data/parsed
"""
import argparse
import os
from pathlib import Path
import pdfplumber


def extract_text_from_pdf(path: Path) -> str:
    text_chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            text_chunks.append(txt)
    return "\n".join(text_chunks)


def process_file(path: Path, out_dir: Path):
    text = ""
    if path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    out_path = out_dir / (path.stem + ".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    for p in in_dir.glob("**/*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}:
            process_file(p, out_dir)

if __name__ == "__main__":
    main()
```

---

## File: src/chunker.py

```python
"""
Chunk text files into overlapping, token-aware chunks using a HuggingFace tokenizer.
Usage:
  python src/chunker.py --input_dir data/parsed --out_dir data/chunks --max_tokens 512 --overlap 128
"""
import argparse
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


def chunk_text_tokenwise(text, tokenizer, max_tokens=512, overlap=128):
    ids = tokenizer.encode(text)
    chunks = []
    i = 0
    while i < len(ids):
        slice_ids = ids[i:i+max_tokens]
        chunk_text = tokenizer.decode(slice_ids, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        if len(ids) <= max_tokens:
            break
        i += max_tokens - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for txt_file in input_dir.glob("**/*.txt"):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text_tokenwise(text, tokenizer, args.max_tokens, args.overlap)
        for i, c in enumerate(chunks):
            out_path = out_dir / f"{txt_file.stem}_chunk_{i}.json"
            metadata = {"source": txt_file.name, "chunk_id": f"{txt_file.stem}_chunk_{i}", "text": c}
            out_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
            manifest.append(str(out_path))
    (out_dir / "manifest.txt").write_text('\n'.join(manifest))
    print(f"Wrote {len(manifest)} chunks to {out_dir}")

if __name__ == "__main__":
    main()
```

---

## File: src/embed\_index.py

```python
"""
Encode chunk files with a sentence-transformers model and build a FAISS index.
Usage:
  python src/embed_index.py --chunks_dir data/chunks --index_path data/index.faiss --embeddings_path data/embeddings.npy
"""
import argparse
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def load_chunks(chunks_dir: Path):
    chunks = []
    files = sorted(chunks_dir.glob("*.json"))
    for f in files:
        obj = json.loads(f.read_text(encoding="utf-8"))
        chunks.append((obj["chunk_id"], obj["text"], obj.get("source")))
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_dir", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    chunks = load_chunks(chunks_dir)
    texts = [c[1] for c in chunks]

    model = SentenceTransformer(args.model)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, args.index_path)
    np.save(args.embeddings_path, embeddings)

    # save simple mapping
    ids = [c[0] for c in chunks]
    with open(chunks_dir / 'id_map.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(ids))
    print(f"Index saved to {args.index_path} (n={len(ids)} vectors)")

if __name__ == "__main__":
    main()
```

---

## Notes & next steps

* These scripts are intentionally minimal and easy to run on CPU for initial testing.
* If you want, I can now:

  * add `retrieve.py` (query -> top-k retrieval) and `generate.py` (RAG prompt assembly), or
  * containerize everything with a Dockerfile, or
  * adapt chunk sizes / models for GPU usage.

Tell me which next step you'd like and I will update the canvas accordingly.
