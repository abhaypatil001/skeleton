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
