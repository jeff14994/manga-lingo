#!/usr/bin/env python3
"""
Add embeddings at extra dimensions for assets that are already ingested.

batch_ingest.py / ingest_video.py create one asset row per file at a single
dimension. Re-ingesting at another dimension would duplicate every asset
(and every file copy). This script instead reads each existing asset's file
and inserts only the missing (asset, dimension) embedding rows.

Usage:
    # Fill 1536 and 3072 for all image + video assets
    python scripts/reembed.py --dim 1536 3072

    # Only images, more parallel requests
    python scripts/reembed.py --dim 3072 --asset-type image --concurrency 6

    # Show what would be embedded
    python scripts/reembed.py --dim 1536 --dry-run
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db import get_db  # noqa: E402
from backend.services.embedding import (  # noqa: E402
    ALLOWED_EMBEDDING_DIMS,
    embed_audio_from_path,
    embed_image_from_path,
    embed_video_from_path,
)

EMBED_FROM_PATH = {
    "image": embed_image_from_path,
    "video": embed_video_from_path,
    "audio": embed_audio_from_path,
}


def find_missing(dim: int, asset_types: list[str]) -> list[tuple[int, str, str]]:
    """Return (asset_id, asset_type, file_path) for assets with no embedding at `dim`."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT a.id, a.asset_type, a.file_path
            FROM assets a
            WHERE a.asset_type = ANY(%s)
              AND NOT EXISTS (
                  SELECT 1 FROM embeddings e
                  WHERE e.asset_id = a.id AND e.embedding_dim = %s
              )
            ORDER BY a.id
            """,
            (asset_types, dim),
        )
        rows = cursor.fetchall()
        cursor.close()
    return rows


def embed_and_store(asset_id: int, asset_type: str, file_path: str, dim: int) -> None:
    """Embed one asset's file at `dim` and insert the embedding row."""
    embedding = EMBED_FROM_PATH[asset_type](str(PROJECT_ROOT / file_path), dim=dim)
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            INSERT INTO embeddings (asset_id, modality, embedding_dim, embedding_{dim})
            VALUES (%s, %s, %s, %s::vector)
            """,
            (asset_id, asset_type, dim, embedding_str),
        )
        cursor.close()


def main():
    parser = argparse.ArgumentParser(description="Add missing embedding dimensions for existing assets")
    parser.add_argument(
        "--dim",
        type=int,
        nargs="+",
        required=True,
        choices=ALLOWED_EMBEDDING_DIMS,
        help="Dimension(s) to fill in, e.g. --dim 1536 3072",
    )
    parser.add_argument(
        "--asset-type",
        nargs="+",
        default=list(EMBED_FROM_PATH),
        choices=list(EMBED_FROM_PATH),
        help="Asset types to process (default: image video audio)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Parallel embedding requests (default: 3; embedding.py also caps in-flight calls)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only count what would be embedded")
    args = parser.parse_args()

    for dim in args.dim:
        missing = find_missing(dim, args.asset_type)
        print(f"\n🔢 {dim}d: {len(missing)} assets missing an embedding")
        if args.dry_run or not missing:
            continue

        start = time.time()
        ok = failed = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(embed_and_store, asset_id, asset_type, file_path, dim): (asset_id, file_path)
                for asset_id, asset_type, file_path in missing
            }
            for i, future in enumerate(as_completed(futures), 1):
                asset_id, file_path = futures[future]
                try:
                    future.result()
                    ok += 1
                    print(f"  [{i:>4}/{len(missing)}] ✅ asset_id={asset_id}")
                except Exception as e:
                    failed += 1
                    print(f"  [{i:>4}/{len(missing)}] ❌ asset_id={asset_id} ({file_path}): {e}")

        print(f"✅ {dim}d done: {ok} embedded, {failed} failed in {time.time() - start:.0f}s")
        if failed:
            print("   Re-run the same command to retry the failures.")


if __name__ == "__main__":
    main()
