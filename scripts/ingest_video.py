#!/usr/bin/env python3
"""
Ingest a video file into Manga-Lingo through the API.

Since Gemini Embedding 2 has a **80-second limit for video with audio**,
this script automatically splits the video into chunks ≤ 80s using ffmpeg,
then ingests each chunk **concurrently** using asyncio + aiohttp.

Prerequisites:
    - ffmpeg must be installed on the machine
    - Backend API must be running
    - PostgreSQL must be running (see README)

Usage:
    # 1. Make sure PostgreSQL is running and .env points at it
    #
    # 2. Make sure the backend is running:
    #      uvicorn backend.main:app --reload --port 8000
    #
    # 3. Ingest the video:
    #      python scripts/ingest_video.py data/video/video_file.mp4
    #
    #    With a manga/anime title:
    #      python scripts/ingest_video.py data/video/video_file.mp4 --title "進擊的巨人 S1E01"
    #
    #    Custom chunk duration (default: 60s, max: 80s):
    #      python scripts/ingest_video.py data/video/video_file.mp4 --chunk-seconds 30
    #
    #    Control concurrency (default: 3):
    #      python scripts/ingest_video.py data/video/video_file.mp4 --concurrency 5
    #
    #    With an existing manga_id:
    #      python scripts/ingest_video.py data/video/video_file.mp4 --manga-id 1

Examples:
    python scripts/ingest_video.py \\
        "data/video/episode01.mp4" \\
        --title "進擊的巨人 S1E01" --concurrency 5
"""

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import aiohttp
except ImportError:
    print("aiohttp is required for async ingestion.")
    print("Install it: pip install aiohttp")
    sys.exit(1)

import requests  # used only for sync health check + manga creation

API_BASE = "http://localhost:8000"

# Gemini Embedding 2 limits
MAX_VIDEO_DURATION_WITH_AUDIO = 80   # seconds
DEFAULT_CHUNK_SECONDS = 15           # 15s chunks for finer-grained video search

# Retry settings
INGEST_MAX_RETRIES = 3
INGEST_RETRY_BASE_DELAY = 3.0  # seconds


def check_api() -> bool:
    """Check if the API is running and DB is connected."""
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=5)
        data = r.json()
        if data.get("db_connected"):
            print("✅ API & Database connected")
            return True
        else:
            print("⚠️  API is up but database is not connected.")
            print("   Make sure PostgreSQL is running and ALLOYDB_* in .env is correct")
            return False
    except requests.ConnectionError:
        print("❌ Cannot reach API at http://localhost:8000")
        print("   Start it with: uvicorn backend.main:app --reload --port 8000")
        return False


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    if shutil.which("ffmpeg") is None:
        print("❌ ffmpeg is not installed. Install it with: sudo apt install ffmpeg")
        return False
    return True


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(video_path),
        ],
        capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def split_video(video_path: Path, chunk_seconds: int, output_dir: Path) -> list[Path]:
    """
    Split a video into chunks of `chunk_seconds` using ffmpeg.
    Returns a list of chunk file paths, ordered by time.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    ext = video_path.suffix

    # Use ffmpeg segment muxer for efficient splitting
    pattern = str(output_dir / f"{stem}_chunk_%03d{ext}")

    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-c", "copy",                      # no re-encoding — fast!
            "-map", "0",
            "-segment_time", str(chunk_seconds),
            "-f", "segment",
            "-reset_timestamps", "1",
            pattern,
        ],
        check=True,
        capture_output=True,
    )

    # Collect and sort the generated chunks
    chunks = sorted(output_dir.glob(f"{stem}_chunk_*{ext}"))
    return chunks


def get_or_create_manga(title: str, language: str = "ja") -> int:
    """Get existing manga by title or create a new one."""
    r = requests.get(f"{API_BASE}/api/manga")
    if r.status_code == 200:
        for m in r.json().get("manga", []):
            if m["title"] == title:
                print(f"📗 Found existing manga: {title} (id={m['id']})")
                return m["id"]

    r = requests.post(f"{API_BASE}/api/manga", json={"title": title, "language": language})
    if r.status_code == 200:
        manga_id = r.json()["manga_id"]
        print(f"📗 Created manga: {title} (id={manga_id})")
        return manga_id
    else:
        print(f"❌ Failed to create manga: {title} — {r.text}")
        sys.exit(1)


async def ingest_chunk_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    chunk_path: Path,
    chunk_index: int,
    total_chunks: int,
    manga_id: int | None = None,
    manga_title: str | None = None,
    embedding_dim: int = 768,
    chunk_start: float | None = None,
    chunk_end: float | None = None,
) -> bool:
    """Upload and ingest a single video chunk through the API with retry."""
    async with semaphore:
        file_size_mb = chunk_path.stat().st_size / (1024 * 1024)

        for attempt in range(INGEST_MAX_RETRIES + 1):
            try:
                data = aiohttp.FormData()
                data.add_field(
                    "file",
                    open(chunk_path, "rb"),
                    filename=chunk_path.name,
                    content_type="video/mp4",
                )
                data.add_field("asset_type", "video")
                data.add_field("embedding_dim", str(embedding_dim))
                if manga_id is not None:
                    data.add_field("manga_id", str(manga_id))
                if manga_title:
                    data.add_field("manga_title", manga_title)
                if chunk_start is not None:
                    data.add_field("chunk_start", str(chunk_start))
                if chunk_end is not None:
                    data.add_field("chunk_end", str(chunk_end))

                start = time.time()
                async with session.post(
                    f"{API_BASE}/api/ingest",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    elapsed = time.time() - start

                    if resp.status == 200:
                        result = await resp.json()
                        retry_note = f" (retry {attempt})" if attempt > 0 else ""
                        print(
                            f"  [{chunk_index:>3}/{total_chunks}] ✅ {chunk_path.name} "
                            f"({file_size_mb:.1f} MB, {elapsed:.1f}s) → asset_id={result['asset_id']}{retry_note}"
                        )
                        return True
                    else:
                        text = await resp.text()
                        if resp.status in (429, 500, 502, 503) and attempt < INGEST_MAX_RETRIES:
                            delay = INGEST_RETRY_BASE_DELAY * (2 ** attempt)
                            print(
                                f"  [{chunk_index:>3}/{total_chunks}] ⚠️  {chunk_path.name} — "
                                f"{resp.status}, retrying in {delay:.0f}s (attempt {attempt + 1}/{INGEST_MAX_RETRIES})"
                            )
                            await asyncio.sleep(delay)
                            continue
                        print(
                            f"  [{chunk_index:>3}/{total_chunks}] ❌ {chunk_path.name} — "
                            f"{resp.status}: {text[:200]}"
                        )
                        return False

            except (aiohttp.ClientError, ConnectionResetError, OSError) as e:
                if attempt < INGEST_MAX_RETRIES:
                    delay = INGEST_RETRY_BASE_DELAY * (2 ** attempt)
                    print(
                        f"  [{chunk_index:>3}/{total_chunks}] ⚠️  {chunk_path.name} — "
                        f"{e}, retrying in {delay:.0f}s (attempt {attempt + 1}/{INGEST_MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)
                else:
                    print(
                        f"  [{chunk_index:>3}/{total_chunks}] ❌ {chunk_path.name} — "
                        f"{e} (exhausted retries)"
                    )
                    return False
            except Exception as e:
                print(f"  [{chunk_index:>3}/{total_chunks}] ❌ {chunk_path.name} — {e}")
                return False

        return False


async def ingest_all_chunks(
    chunks: list[Path],
    concurrency: int,
    manga_id: int | None = None,
    manga_title: str | None = None,
    embedding_dim: int = 768,
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
    total_duration: float = 0.0,
) -> tuple[int, int]:
    """
    Ingest all video chunks concurrently. Returns (success, failed).

    Uses TCPConnector with connection limits and a stagger delay to
    prevent thundering-herd connection resets (same pattern as batch_ingest.py).
    
    Each chunk gets chunk_start/chunk_end metadata calculated from its
    index and the chunk duration, so video search results can display
    the time range (e.g. "0:01:15–0:01:30").
    """
    semaphore = asyncio.Semaphore(concurrency)
    total = len(chunks)

    # Stagger delay between launching tasks (seconds)
    STAGGER_DELAY = 0.2

    connector = aiohttp.TCPConnector(
        limit=concurrency,
        limit_per_host=concurrency,
        keepalive_timeout=30,
        enable_cleanup_closed=True,
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i, chunk in enumerate(chunks, 1):
            # Calculate time range for this chunk
            c_start = (i - 1) * chunk_seconds
            c_end = min(i * chunk_seconds, total_duration) if total_duration > 0 else i * chunk_seconds
            tasks.append(asyncio.ensure_future(
                ingest_chunk_async(
                    session, semaphore, chunk, i, total,
                    manga_id=manga_id, manga_title=manga_title,
                    embedding_dim=embedding_dim,
                    chunk_start=float(c_start),
                    chunk_end=float(c_end),
                )
            ))
            if i < len(chunks):
                await asyncio.sleep(STAGGER_DELAY)

        results = await asyncio.gather(*tasks)

    success = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)
    return success, failed


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a video into Manga-Lingo (auto-splits into ≤80s chunks, async embedding)"
    )
    parser.add_argument("video", type=str, help="Path to the video file (e.g. data/video/xxx.mp4)")
    parser.add_argument("--title", type=str, default=None,
                        help="Manga/anime title (will create or reuse manga entry)")
    parser.add_argument("--manga-id", type=int, default=None,
                        help="Existing manga ID to associate with")
    parser.add_argument("--language", type=str, default="ja",
                        help="Language code (default: ja)")
    parser.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS,
                        help=f"Chunk duration in seconds (default: {DEFAULT_CHUNK_SECONDS}, max: {MAX_VIDEO_DURATION_WITH_AUDIO})")
    parser.add_argument("--concurrency", "-c", type=int, default=10,
                        help="Number of concurrent API requests (default: 10)")
    parser.add_argument("--embedding-dim", type=int, default=768,
                        choices=[768, 1536, 3072],
                        help="Embedding dimension (default: 768). Higher = better quality, more cost/storage.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)

    if args.chunk_seconds > MAX_VIDEO_DURATION_WITH_AUDIO:
        print(f"⚠️  Chunk duration {args.chunk_seconds}s exceeds Gemini limit of {MAX_VIDEO_DURATION_WITH_AUDIO}s.")
        print(f"   Setting to {MAX_VIDEO_DURATION_WITH_AUDIO}s.")
        args.chunk_seconds = MAX_VIDEO_DURATION_WITH_AUDIO

    print("=" * 60)
    print("📹 Manga-Lingo — Video Ingestion (Async)")
    print("=" * 60)
    print()

    # Check prerequisites
    print("🔍 Checking prerequisites...")
    if not check_ffmpeg():
        sys.exit(1)
    if not check_api():
        sys.exit(1)
    print()

    # Get video info
    duration = get_video_duration(video_path)
    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    num_chunks = max(1, int(duration // args.chunk_seconds) + (1 if duration % args.chunk_seconds > 0 else 0))

    print(f"  Video:       {video_path.name}")
    print(f"  Size:        {file_size_mb:.1f} MB")
    print(f"  Duration:    {duration:.0f}s ({duration/60:.1f} min)")
    print(f"  Chunks:      {num_chunks} × {args.chunk_seconds}s")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Embed dim:   {args.embedding_dim}")
    print()

    # Handle manga association
    manga_id = args.manga_id
    manga_title = args.title

    if args.title and manga_id is None:
        manga_id = get_or_create_manga(args.title, args.language)
        print()

    # Split video if needed
    if duration <= args.chunk_seconds:
        print("📹 Video is short enough — ingesting directly (no splitting needed)")
        chunks = [video_path]
        tmp_dir = None
    else:
        print(f"✂️  Splitting video into {num_chunks} chunks of ≤{args.chunk_seconds}s...")
        tmp_dir = tempfile.mkdtemp(prefix="manga_lingo_chunks_")
        tmp_path = Path(tmp_dir)
        chunks = split_video(video_path, args.chunk_seconds, tmp_path)
        print(f"   Created {len(chunks)} chunks in {tmp_dir}")
        print()

    # Ingest all chunks concurrently
    print(f"🚀 Ingesting {len(chunks)} chunk(s) with concurrency={args.concurrency}...")
    print(f"   Each chunk will be embedded via Gemini Embedding 2")
    print()

    start_time = time.time()
    success, failed = asyncio.run(
        ingest_all_chunks(
            chunks, args.concurrency,
            manga_id=manga_id, manga_title=manga_title,
            embedding_dim=args.embedding_dim,
            chunk_seconds=args.chunk_seconds,
            total_duration=duration,
        )
    )
    elapsed = time.time() - start_time

    # Cleanup temp chunks
    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print()
    print("=" * 60)
    print(f"✅ Done! {success} chunks ingested, {failed} failed ({elapsed:.1f}s total)")
    if success > 0:
        print(f"   Avg: {elapsed/success:.1f}s per chunk (wall clock: {elapsed:.1f}s for {success} chunks)")
    if failed > 0:
        print(f"   ⚠️  {failed} chunks failed — check the API logs for details")
    print()
    print("🎉 You can now search for this video in the Streamlit UI!")
    print("=" * 60)


if __name__ == "__main__":
    main()
