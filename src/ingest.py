"""
Ingest raw PDF and TXT files from a directory and write cleaned plain-text files

Usage:
  python src/ingest.py --input_dir data/raw --out_dir data/parsed
"""
import argparse
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
    if path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (path.stem + ".txt")
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
