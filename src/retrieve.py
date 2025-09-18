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
