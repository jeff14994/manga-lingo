#!/usr/bin/env python3
"""
Ingest a PDF document into Manga-Lingo via the FastAPI /api/ingest/pdf endpoint.

Usage:
    python scripts/ingest_pdf.py "data/docs/進級的巨人研究.pdf" --embedding-dim 768
    python scripts/ingest_pdf.py "data/docs/進級的巨人研究.pdf" --embedding-dim 1536 --manga-title "進擊的巨人研究"
    python scripts/ingest_pdf.py "data/docs/進級的巨人研究.pdf" --embedding-dim 3072 --manga-id 1
"""

import argparse
import sys
from pathlib import Path

import requests

API_BASE = "http://localhost:8000"


def ingest_pdf(
    pdf_path: str,
    embedding_dim: int = 768,
    manga_id: int | None = None,
    manga_title: str | None = None,
    concurrency: int = 5,
):
    path = Path(pdf_path)
    if not path.exists():
        print(f"[ERROR] File not found: {pdf_path}")
        sys.exit(1)

    if not path.suffix.lower() == ".pdf":
        print(f"[ERROR] Not a PDF file: {pdf_path}")
        sys.exit(1)

    print(f"[PDF] Ingesting: {path.name}")
    print(f"[PDF] Embedding dimension: {embedding_dim}")
    print(f"[PDF] File size: {path.stat().st_size / 1024:.1f} KB")

    files = {"file": (path.name, path.read_bytes(), "application/pdf")}
    data = {"embedding_dim": str(embedding_dim), "max_workers": str(concurrency)}
    if manga_id is not None:
        data["manga_id"] = str(manga_id)
    if manga_title:
        data["manga_title"] = manga_title

    try:
        r = requests.post(f"{API_BASE}/api/ingest/pdf", files=files, data=data, timeout=300)
        if r.status_code == 200:
            result = r.json()
            pages = result.get("pages_ingested", 0)
            print(f"[PDF] ✅ Successfully ingested {pages} pages")
            for page in result.get("pages", []):
                text_preview = page.get("dialogue_text", "")[:80]
                print(f"  Page {page['page_number']}: asset_id={page['asset_id']} — {text_preview}...")
        else:
            print(f"[PDF] ❌ Error {r.status_code}: {r.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot connect to {API_BASE}. Is the backend running?")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF document into Manga-Lingo")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--embedding-dim", type=int, default=768, choices=[768, 1536, 3072])
    parser.add_argument("--manga-id", type=int, default=None, help="Manga ID to associate with")
    parser.add_argument("--manga-title", type=str, default=None, help="Title for display")
    parser.add_argument("--concurrency", "-c", type=int, default=5,
                        help="Number of concurrent embedding workers (default: 5)")
    args = parser.parse_args()

    ingest_pdf(
        pdf_path=args.pdf_path,
        embedding_dim=args.embedding_dim,
        manga_id=args.manga_id,
        manga_title=args.manga_title,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
