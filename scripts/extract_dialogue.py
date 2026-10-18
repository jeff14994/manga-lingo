#!/usr/bin/env python3
"""
Extract Japanese dialogue for study: speech bubbles from manga pages and
timestamped transcripts from video chunks (Gemini), then store lines,
furigana-ready words and JLPT levels in the database.

Only assets whose manga language is 'ja' are processed. Assets that already
have lines are skipped unless --redo is given.

Usage:
    # Manga pages and video chunks
    python scripts/extract_dialogue.py

    # Only video, one title
    python scripts/extract_dialogue.py --asset-type video --title "進擊的巨人 S1E01"

    # Try a few pages first
    python scripts/extract_dialogue.py --asset-type image --limit 3
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db import get_db  # noqa: E402
from backend.services import study  # noqa: E402
from backend.services.dialogue import extract_manga_lines, transcribe_video_chunk  # noqa: E402


def find_assets(asset_types: list[str], title: str | None, redo: bool, limit: int | None) -> list[tuple]:
    """Return (asset_id, asset_type, file_path, chunk_start) to process."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT a.id, a.asset_type, a.file_path, a.chunk_start
            FROM assets a JOIN manga m ON m.id = a.manga_id
            WHERE m.language = 'ja'
              AND a.asset_type = ANY(%s)
              AND (%s::text IS NULL OR m.title = %s)
              AND (%s OR NOT EXISTS (SELECT 1 FROM dialogue_lines l WHERE l.asset_id = a.id))
            ORDER BY a.manga_id, a.page_number NULLS LAST, a.chunk_start NULLS LAST, a.id
            LIMIT %s
            """,
            (asset_types, title, title, redo, limit),
        )
        rows = cursor.fetchall()
        cursor.close()
    return rows


def process(asset_id: int, asset_type: str, file_path: str, chunk_start: float | None) -> int:
    data = (PROJECT_ROOT / file_path).read_bytes()
    if asset_type == "image":
        lines = extract_manga_lines(data)
        return study.save_lines(asset_id, "manga", lines)
    lines = transcribe_video_chunk(data, chunk_start or 0.0)
    return study.save_lines(asset_id, "video", lines)


def main():
    parser = argparse.ArgumentParser(description="Extract Japanese dialogue lines with Gemini")
    parser.add_argument("--asset-type", nargs="+", default=["image", "video"], choices=["image", "video"])
    parser.add_argument("--title", type=str, default=None, help="Only this manga/video title")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N assets")
    parser.add_argument("--redo", action="store_true", help="Re-extract assets that already have lines")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel Gemini requests (default: 4)")
    args = parser.parse_args()

    assets = find_assets(args.asset_type, args.title, args.redo, args.limit)
    print(f"📝 {len(assets)} assets to process")
    if not assets:
        return

    start = time.time()
    total_lines = failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(process, *row): row for row in assets}
        for i, future in enumerate(as_completed(futures), 1):
            asset_id, asset_type, file_path, _ = futures[future]
            try:
                n = future.result()
                total_lines += n
                print(f"  [{i:>4}/{len(assets)}] ✅ {asset_type} asset_id={asset_id}: {n} lines")
            except Exception as e:
                failed += 1
                print(f"  [{i:>4}/{len(assets)}] ❌ {asset_type} asset_id={asset_id} ({file_path}): {e}")

    print(f"\n✅ Done: {total_lines} lines from {len(assets) - failed} assets, {failed} failed "
          f"in {time.time() - start:.0f}s")
    if failed:
        print("   Re-run the same command to retry the failures.")


if __name__ == "__main__":
    main()
