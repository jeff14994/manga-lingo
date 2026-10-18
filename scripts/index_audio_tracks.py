#!/usr/bin/env python3
"""
Index the audio track of every video chunk as its own audio asset.

Audio → video search compares an audio clip with stored embeddings. A video
chunk's embedding is dominated by its visuals, so music-heavy scenes (e.g. an
opening montage) can't be found by their sound. Embedding each chunk's audio
track separately fixes that: search/audio-to-video matches the audio tracks
and returns the video chunk with the same (manga_id, chunk_start).

Usage:
    python scripts/index_audio_tracks.py                  # all video chunks, 768d
    python scripts/index_audio_tracks.py --title "進擊的巨人 S1E01"
    python scripts/reembed.py --dim 1536 3072 --asset-type audio   # other dims
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db import get_db  # noqa: E402
from backend.services.ingest import ingest_asset  # noqa: E402


def find_chunks(title: str | None) -> list[dict]:
    """Video chunks that don't have an audio-track asset yet."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT v.id, v.manga_id, m.title, v.file_path, v.chunk_start, v.chunk_end,
                   v.dialogue_text, v.translation_en
            FROM assets v JOIN manga m ON m.id = v.manga_id
            WHERE v.asset_type = 'video' AND v.chunk_start IS NOT NULL
              AND (%s::text IS NULL OR m.title = %s)
              AND NOT EXISTS (
                  SELECT 1 FROM assets a
                  WHERE a.asset_type = 'audio' AND a.manga_id = v.manga_id
                    AND a.chunk_start = v.chunk_start
              )
            ORDER BY v.manga_id, v.chunk_start
            """,
            (title, title),
        )
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
        cursor.close()
    return rows


def extract_audio(video_path: Path) -> bytes:
    """Mono 96 kbps mp3 of the chunk's audio track (piped, no temp file)."""
    return subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(video_path), "-vn", "-ac", "1",
         "-ar", "22050", "-b:a", "96k", "-f", "mp3", "pipe:1"],
        check=True, capture_output=True,
    ).stdout


def index_one(chunk: dict, dim: int) -> int:
    audio = extract_audio(PROJECT_ROOT / chunk["file_path"])
    result = ingest_asset(
        file_bytes=audio,
        filename=f"{Path(chunk['file_path']).stem}.mp3",
        asset_type="audio",
        manga_id=chunk["manga_id"],
        manga_title=chunk["title"],
        dialogue_text=chunk["dialogue_text"],
        translation_en=chunk["translation_en"],
        embedding_dim=dim,
        chunk_start=chunk["chunk_start"],
        chunk_end=chunk["chunk_end"],
    )
    return result["asset_id"]


def main():
    parser = argparse.ArgumentParser(description="Index video chunks' audio tracks as audio assets")
    parser.add_argument("--title", type=str, default=None, help="Only this video title")
    parser.add_argument("--embedding-dim", type=int, default=768, choices=[768, 1536, 3072])
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    chunks = find_chunks(args.title)
    print(f"🎧 {len(chunks)} video chunks without an audio track")
    if not chunks:
        return

    start, ok, failed = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(index_one, c, args.embedding_dim): c for c in chunks}
        for i, future in enumerate(as_completed(futures), 1):
            c = futures[future]
            label = f"{c['title']} {int(c['chunk_start']) // 60}:{int(c['chunk_start']) % 60:02d}"
            try:
                asset_id = future.result()
                ok += 1
                print(f"  [{i:>3}/{len(chunks)}] ✅ {label} → asset_id={asset_id}")
            except Exception as e:
                failed += 1
                print(f"  [{i:>3}/{len(chunks)}] ❌ {label}: {e}")

    print(f"\n✅ Done: {ok} indexed, {failed} failed in {time.time() - start:.0f}s")
    if failed:
        print("   Re-run the same command to retry the failures.")


if __name__ == "__main__":
    main()
