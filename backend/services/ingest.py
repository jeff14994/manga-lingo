"""
Ingestion service: saves files locally, generates embeddings, stores in PostgreSQL (pgvector).
"""

import os
import uuid
from pathlib import Path

from backend.config import DATA_DIR
from backend.db import get_db
from backend.services.embedding import (
    embed_image,
    embed_audio,
    embed_video,
    embed_document,
    DEFAULT_EMBEDDING_DIM,
    ALLOWED_EMBEDDING_DIMS,
)

# Map asset types to subdirectories
ASSET_DIRS = {
    "image": "images",
    "audio": "audio",
    "video": "video",
    "pdf": "docs",
}

# Map asset types to embedding functions
# For "pdf", embedding is text-based (handled specially in ingest_asset)
EMBED_FUNCS = {
    "image": embed_image,
    "audio": embed_audio,
    "video": embed_video,
}


def _sanitize_dirname(name: str) -> str:
    """Sanitize a string for use as a directory name."""
    # Replace spaces with underscores, keep unicode chars (manga titles)
    return name.replace("/", "_").replace("\\", "_").strip()


def save_file(
    file_bytes: bytes,
    asset_type: str,
    filename: str,
    manga_title: str | None = None,
) -> str:
    """
    Save a file to the local data directory, organized by manga title.
    
    With manga_title:  data/images/进击的巨人_第01卷/abc123.webp
    Without:           data/images/abc123.webp
    
    Returns the relative file path.
    """
    subdir = ASSET_DIRS.get(asset_type)
    if not subdir:
        raise ValueError(f"Unknown asset_type: {asset_type}")

    # Build destination directory, optionally nested by manga title
    base_dir = DATA_DIR / subdir
    if manga_title:
        base_dir = base_dir / _sanitize_dirname(manga_title)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename to avoid collisions
    ext = Path(filename).suffix or _default_ext(asset_type)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest = base_dir / unique_name
    dest.write_bytes(file_bytes)

    # Return path relative to project root
    return str(dest.relative_to(DATA_DIR.parent))


def _default_ext(asset_type: str) -> str:
    """Default file extension for each asset type."""
    return {
        "image": ".png",
        "audio": ".mp3",
        "video": ".mp4",
    }.get(asset_type, ".bin")


def create_manga(title: str, language: str = "ja") -> int:
    """Create a manga entry and return its ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO manga (title, language) VALUES (%s, %s) RETURNING id",
            (title, language),
        )
        manga_id = cursor.fetchone()[0]
        cursor.close()
    return manga_id


def list_manga() -> list[dict]:
    """List all manga entries."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, language, created_at FROM manga ORDER BY created_at DESC")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()
    return [dict(zip(columns, row)) for row in rows]


def _embedding_column(dim: int) -> str:
    """Return the DB column name for the given embedding dimension."""
    return f"embedding_{dim}"


def ingest_asset(
    file_bytes: bytes,
    filename: str,
    asset_type: str,
    manga_id: int | None = None,
    manga_title: str | None = None,
    dialogue_text: str | None = None,
    translation_en: str | None = None,
    page_number: int | None = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    chunk_start: float | None = None,
    chunk_end: float | None = None,
) -> dict:
    """
    Full ingestion pipeline:
    1. Save file to local filesystem (organized by manga_title if provided)
    2. Generate embedding via Gemini Embedding 2 at the requested dimension
    3. Insert asset record into PostgreSQL
    4. Insert embedding record (dimension-specific column)
    
    Args:
        embedding_dim: Embedding output dimension (768, 1536, or 3072).
    
    Returns the created asset info.
    """
    if embedding_dim not in ALLOWED_EMBEDDING_DIMS:
        raise ValueError(f"Invalid embedding_dim {embedding_dim}. Allowed: {ALLOWED_EMBEDDING_DIMS}")

    # Auto-extract page number from filename if not provided
    # e.g. '001.webp' → 1, '042.png' → 42
    if page_number is None:
        stem = Path(filename).stem
        if stem.isdigit():
            page_number = int(stem)

    # 1. Save file
    file_path = save_file(file_bytes, asset_type, filename, manga_title=manga_title)

    # 2. Generate embedding at the requested dimension
    embed_func = EMBED_FUNCS.get(asset_type)
    if not embed_func:
        raise ValueError(f"Cannot embed asset_type: {asset_type}")

    embedding = embed_func(file_bytes, dim=embedding_dim)
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    # 3 & 4. Insert into DB
    col = _embedding_column(embedding_dim)

    with get_db() as conn:
        cursor = conn.cursor()

        # Insert asset
        cursor.execute(
            """
            INSERT INTO assets (manga_id, asset_type, file_path, page_number, dialogue_text, translation_en, chunk_start, chunk_end)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (manga_id, asset_type, file_path, page_number, dialogue_text, translation_en, chunk_start, chunk_end),
        )
        asset_id = cursor.fetchone()[0]

        # Insert embedding into the dimension-specific column
        cursor.execute(
            f"""
            INSERT INTO embeddings (asset_id, modality, embedding_dim, {col})
            VALUES (%s, %s, %s, %s::vector)
            """,
            (asset_id, asset_type, embedding_dim, embedding_str),
        )

        cursor.close()

    return {
        "asset_id": asset_id,
        "file_path": file_path,
        "asset_type": asset_type,
        "manga_id": manga_id,
        "page_number": page_number,
        "dialogue_text": dialogue_text,
        "translation_en": translation_en,
        "embedding_dim": embedding_dim,
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
    }


def _embed_and_store_pdf_page(
    page_num: int,
    text: str,
    file_path: str,
    manga_id: int | None,
    embedding_dim: int,
    col: str,
) -> dict | None:
    """Embed a single PDF page and store in DB. Returns page info dict or None on failure."""
    try:
        # Truncate very long pages to avoid API limits (~10k chars)
        if len(text) > 10000:
            text = text[:10000]

        print(f"[PDF] Page {page_num}: {len(text)} chars — embedding...")

        embedding = embed_document(text, dim=embedding_dim)
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO assets (manga_id, asset_type, file_path, page_number, dialogue_text)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (manga_id, "pdf", file_path, page_num, text[:500]),
            )
            asset_id = cursor.fetchone()[0]

            cursor.execute(
                f"""
                INSERT INTO embeddings (asset_id, modality, embedding_dim, {col})
                VALUES (%s, %s, %s, %s::vector)
                """,
                (asset_id, "pdf", embedding_dim, embedding_str),
            )
            cursor.close()

        print(f"[PDF] Page {page_num}: ✅ asset_id={asset_id}")
        return {
            "asset_id": asset_id,
            "file_path": file_path,
            "asset_type": "pdf",
            "page_number": page_num,
            "dialogue_text": text[:200],
            "embedding_dim": embedding_dim,
        }
    except Exception as e:
        print(f"[PDF] Page {page_num}: ❌ {e}")
        return None


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    manga_id: int | None = None,
    manga_title: str | None = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    max_workers: int = 5,
) -> list[dict]:
    """
    Ingest a PDF document with concurrent page embedding:
    1. Save the PDF file to data/docs/
    2. Extract text from each page using PyMuPDF (fitz)
    3. Embed pages concurrently via ThreadPoolExecutor
    4. Store one asset row per page (asset_type='pdf', page_number=page)
    5. Store one embedding row per page (modality='pdf')

    Returns a list of ingested page info dicts (sorted by page number).
    """
    import fitz  # PyMuPDF
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if embedding_dim not in ALLOWED_EMBEDDING_DIMS:
        raise ValueError(f"Invalid embedding_dim {embedding_dim}. Allowed: {ALLOWED_EMBEDDING_DIMS}")

    # 1. Save PDF file
    file_path = save_file(file_bytes, "pdf", filename, manga_title=manga_title)

    # 2. Extract text per page
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    col = _embedding_column(embedding_dim)

    # Collect pages with text
    pages_to_embed: list[tuple[int, str]] = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text").strip()
        page_num = page_idx + 1

        if not text:
            print(f"[PDF] Page {page_num}: no text, skipping")
            continue

        pages_to_embed.append((page_num, text))

    doc.close()

    print(f"[PDF] {len(pages_to_embed)} pages with text, embedding with {max_workers} workers...")

    # 3-5. Embed and store pages concurrently
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pdf_embed") as pool:
        futures = {
            pool.submit(
                _embed_and_store_pdf_page,
                page_num, text, file_path, manga_id, embedding_dim, col,
            ): page_num
            for page_num, text in pages_to_embed
        }

        for future in as_completed(futures):
            page_num = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"[PDF] Page {page_num}: ❌ {e}")

    # Sort by page number for consistent output
    results.sort(key=lambda x: x["page_number"])
    print(f"[PDF] ✅ Done: {len(results)}/{len(pages_to_embed)} pages ingested")
    return results
