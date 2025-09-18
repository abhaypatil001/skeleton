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
