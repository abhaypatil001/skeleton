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
