"""
Manga-Lingo MVP — FastAPI Backend
"""

import asyncio
import mimetypes
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from backend.config import DATA_DIR
from backend.db import test_connection, close_pool
from backend.services import search, ingest, study
from backend.services.embedding import DEFAULT_EMBEDDING_DIM, ALLOWED_EMBEDDING_DIMS

app = FastAPI(title="Manga-Lingo API", version="0.1.0")

# Thread pool for running synchronous, blocking work (embedding + DB)
# without starving the async event loop.  Size matches the DB pool max.
_ingest_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ingest")


@app.on_event("shutdown")
def _shutdown():
    """Clean up resources on server shutdown."""
    _ingest_pool.shutdown(wait=False)
    close_pool()

# CORS — only the Streamlit frontend. A wildcard would let any web page the
# user visits read API responses (and files) from localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request/Response Models ──────────────────────────────────────────────

class TextSearchRequest(BaseModel):
    query: str
    limit: int = 10
    embedding_dim: int = 768   # 768, 1536, or 3072


class CreateMangaRequest(BaseModel):
    title: str
    language: str = "ja"


# ── File Serving (handles Unicode paths reliably) ────────────────────────

PROJECT_ROOT = DATA_DIR.parent


def _media_path(relative_path: str) -> Path:
    """
    Resolve a stored media path and refuse anything outside data/ — e.g.
    ".env", source files, or "../sibling-dir" tricks.
    """
    full_path = (PROJECT_ROOT / relative_path).resolve()
    if not full_path.is_relative_to(DATA_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")
    return full_path


@app.get("/api/file/{asset_id}")
def serve_file_by_id(asset_id: int):
    """
    Serve a file by asset ID — looks up file_path from DB.
    NOTE: Prefer /api/serve/{path} for concurrent image loading to avoid
    DB connection pool exhaustion. This endpoint is kept for backward compat.
    """
    from backend.db import get_db

    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path, asset_type FROM assets WHERE id = %s", (asset_id,))
            row = cursor.fetchone()
            cursor.close()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database unavailable: {e}",
        )

    if not row:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")

    file_path_str, asset_type = row
    full_path = _media_path(file_path_str)

    if not full_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found on disk: {file_path_str}",
        )

    # Guess media type
    media_type = mimetypes.guess_type(str(full_path))[0]
    if not media_type:
        type_map = {"image": "image/webp", "audio": "audio/mpeg", "video": "video/mp4", "pdf": "application/pdf"}
        media_type = type_map.get(asset_type, "application/octet-stream")

    return FileResponse(full_path, media_type=media_type)


@app.get("/api/serve/{file_path:path}")
def serve_file_by_path(file_path: str):
    """
    Serve a file by its relative path — handles Unicode (Chinese/Japanese/Korean)
    directory and file names reliably.

    This is a fallback for when asset_id is not available. The path is
    URL-decoded automatically by FastAPI, so non-ASCII characters work correctly.
    """
    # URL-decode the path (FastAPI does this automatically for path params,
    # but double-decode in case the client double-encoded)
    decoded_path = urllib.parse.unquote(file_path)
    full_path = _media_path(decoded_path)

    if not full_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {decoded_path}",
        )

    media_type = mimetypes.guess_type(str(full_path))[0] or "application/octet-stream"
    return FileResponse(full_path, media_type=media_type)


# ── Health Check ─────────────────────────────────────────────────────────

@app.get("/api/ping")
def ping():
    """Simple ping — no DB check."""
    return {"status": "ok"}


@app.get("/api/health")
def health():
    """Full health check including DB. May be slow if DB is unreachable."""
    db_ok = test_connection()
    return {"status": "ok" if db_ok else "db_error", "db_connected": db_ok}


# ── Search Endpoints (Step 1: Text → X) ─────────────────────────────────

@app.post("/api/search/text-to-image")
def search_text_to_image(req: TextSearchRequest):
    try:
        results = search.text_to_image(req.query, req.limit, embedding_dim=req.embedding_dim)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/text-to-audio")
def search_text_to_audio(req: TextSearchRequest):
    try:
        results = search.text_to_audio(req.query, req.limit, embedding_dim=req.embedding_dim)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/text-to-video")
def search_text_to_video(req: TextSearchRequest):
    try:
        results = search.text_to_video(req.query, req.limit, embedding_dim=req.embedding_dim)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/text-to-pdf")
def search_text_to_pdf(req: TextSearchRequest):
    try:
        results = search.text_to_pdf(req.query, req.limit, embedding_dim=req.embedding_dim)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Search Endpoints (Step 2: Image → X) ────────────────────────────────

@app.post("/api/search/image-to-image")
async def search_image_to_image(
    file: UploadFile = File(...),
    limit: int = Form(10),
    embedding_dim: int = Form(DEFAULT_EMBEDDING_DIM),
):
    try:
        image_bytes = await file.read()
        results = search.image_to_image(image_bytes, limit, embedding_dim=embedding_dim)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/image-to-video")
async def search_image_to_video(
    file: UploadFile = File(...),
    limit: int = Form(10),
    embedding_dim: int = Form(DEFAULT_EMBEDDING_DIM),
):
    try:
        image_bytes = await file.read()
        results = search.image_to_video(image_bytes, limit, embedding_dim=embedding_dim)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Search Endpoints (Step 3: Audio → X) ────────────────────────────────

@app.post("/api/search/audio-to-video")
async def search_audio_to_video(
    file: UploadFile = File(...),
    limit: int = Form(10),
    embedding_dim: int = Form(DEFAULT_EMBEDDING_DIM),
):
    """Find anime scenes from a short audio clip (a line of dialogue, the opening song, ...)."""
    mime_type = (
        file.content_type
        if file.content_type and file.content_type.startswith("audio/")
        else mimetypes.guess_type(file.filename or "")[0] or "audio/mpeg"
    )
    try:
        audio_bytes = await file.read()
        results = search.audio_to_video(audio_bytes, limit, embedding_dim=embedding_dim, mime_type=mime_type)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Ingestion Endpoints ─────────────────────────────────────────────────

@app.post("/api/manga")
def create_manga(req: CreateMangaRequest):
    try:
        manga_id = ingest.create_manga(req.title, req.language)
        return {"manga_id": manga_id, "title": req.title, "language": req.language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/manga")
def list_manga():
    try:
        manga_list = ingest.list_manga()
        return {"manga": manga_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest")
async def ingest_asset(
    file: UploadFile = File(...),
    asset_type: str = Form(...),
    manga_id: Optional[int] = Form(None),
    manga_title: Optional[str] = Form(None),
    dialogue_text: Optional[str] = Form(None),
    translation_en: Optional[str] = Form(None),
    page_number: Optional[int] = Form(None),
    embedding_dim: int = Form(DEFAULT_EMBEDDING_DIM),
    chunk_start: Optional[float] = Form(None),
    chunk_end: Optional[float] = Form(None),
):
    if asset_type not in ("image", "audio", "video", "pdf"):
        raise HTTPException(status_code=400, detail="asset_type must be 'image', 'audio', 'video', or 'pdf'")
    if embedding_dim not in ALLOWED_EMBEDDING_DIMS:
        raise HTTPException(
            status_code=400,
            detail=f"embedding_dim must be one of {ALLOWED_EMBEDDING_DIMS}",
        )

    try:
        file_bytes = await file.read()

        # Run the blocking ingest pipeline (Gemini API call + DB write)
        # in a thread pool so it doesn't block the async event loop.
        # This allows uvicorn to keep accepting new requests while
        # previous ones are waiting on the embedding API.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _ingest_pool,
            lambda: ingest.ingest_asset(
                file_bytes=file_bytes,
                filename=file.filename or "upload",
                asset_type=asset_type,
                manga_id=manga_id,
                manga_title=manga_title,
                dialogue_text=dialogue_text,
                translation_en=translation_en,
                page_number=page_number,
                embedding_dim=embedding_dim,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
            ),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    manga_id: Optional[int] = Form(None),
    manga_title: Optional[str] = Form(None),
    embedding_dim: int = Form(DEFAULT_EMBEDDING_DIM),
    max_workers: int = Form(5),
):
    """Ingest a PDF document — extracts text per page and embeds each page concurrently."""
    if embedding_dim not in ALLOWED_EMBEDDING_DIMS:
        raise HTTPException(
            status_code=400,
            detail=f"embedding_dim must be one of {ALLOWED_EMBEDDING_DIMS}",
        )
    # Clamp max_workers to a sane range
    max_workers = max(1, min(max_workers, 20))

    try:
        file_bytes = await file.read()
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            _ingest_pool,
            lambda: ingest.ingest_pdf(
                file_bytes=file_bytes,
                filename=file.filename or "upload.pdf",
                manga_id=manga_id,
                manga_title=manga_title,
                embedding_dim=embedding_dim,
                max_workers=max_workers,
            ),
        )
        return {"pages_ingested": len(results), "pages": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Study Endpoints (Japanese reading + review) ──────────────────────────

class ReviewAddRequest(BaseModel):
    lemma: str
    example_line_id: Optional[int] = None


class ReviewGradeRequest(BaseModel):
    grade: str   # again, hard, good, easy


@app.get("/api/manga/{manga_id}/pages")
def list_manga_pages(manga_id: int):
    try:
        return {"pages": study.list_pages(manga_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/assets/{asset_id}/lines")
def get_asset_lines(asset_id: int):
    """Dialogue lines of a page or video chunk, with furigana and word breakdown."""
    try:
        result = study.get_asset_lines(asset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not result:
        raise HTTPException(status_code=404, detail="Asset not found")
    return result


@app.get("/api/words/{lemma}/occurrences")
def word_occurrences(lemma: str, source: Optional[str] = None, limit: int = 20):
    """Where a word appears; source='video' gives anime moments with playback offsets."""
    if source not in (None, "manga", "video"):
        raise HTTPException(status_code=400, detail="source must be 'manga' or 'video'")
    try:
        return {"occurrences": study.word_occurrences(lemma, source, limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/review/items")
def add_review_item(req: ReviewAddRequest):
    try:
        return study.add_review_item(req.lemma, req.example_line_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/review/due")
def review_due(limit: int = 20):
    try:
        return {"items": study.due_items(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/review/{item_id}/grade")
def grade_review_item(item_id: int, req: ReviewGradeRequest):
    if req.grade not in study.GRADES:
        raise HTTPException(status_code=400, detail=f"grade must be one of {list(study.GRADES)}")
    try:
        return study.grade_item(item_id, req.grade)
    except KeyError:
        raise HTTPException(status_code=404, detail="Review item not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/assets/{asset_id}/scenes")
def asset_scenes(asset_id: int, limit: int = 3, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
    """
    Anime scenes matching a manga page (by its stored embedding), each with
    its Japanese transcript lines (furigana + words) for listening practice.
    """
    if embedding_dim not in ALLOWED_EMBEDDING_DIMS:
        raise HTTPException(status_code=400, detail=f"embedding_dim must be one of {ALLOWED_EMBEDDING_DIMS}")
    try:
        scenes = search.asset_to_video(asset_id, limit, embedding_dim=embedding_dim)
        for scene in scenes:
            scene["lines"] = study.get_asset_lines(scene["asset_id"]).get("lines", [])
        return {"scenes": scenes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
