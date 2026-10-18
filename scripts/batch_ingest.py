#!/usr/bin/env python3
"""
Manga-Lingo: Async Batch Ingest from Downloads

Scans the downloads/ folder for manga images and ingests them
through the API concurrently using asyncio + aiohttp.

The manga title auto-includes the volume/episode from the folder name:
  downloads/进击的巨人/第01卷/  →  title = "进击的巨人 第01卷"

Usage:
    # 1. Make sure PostgreSQL is running and .env points at it
    #
    # 2. Make sure the backend is running:
    #      uvicorn backend.main:app --reload --port 8000
    #
    # 3. Run this script:
    #      python scripts/batch_ingest.py downloads/进击的巨人/第01卷
    #
    #    Or ingest all volumes at once:
    #      python scripts/batch_ingest.py downloads/进击的巨人
    #
    #    Control concurrency (default 5):
    #      python scripts/batch_ingest.py downloads/进击的巨人/第01卷 --concurrency 10
    #
    #    Resume / retry failed images (skips already-ingested):
    #      python scripts/batch_ingest.py downloads/进击的巨人/第01卷 --resume
"""

import argparse
import asyncio
import json
import logging
import mimetypes
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import aiohttp
except ImportError:
    print("aiohttp is required for async ingestion.")
    print("Install it: pip install aiohttp")
    sys.exit(1)

import requests  # used only for sync health check + manga creation

API_BASE = "http://localhost:8000"

# ── Logging Setup ────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_filename = LOG_DIR / f"batch_ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("batch_ingest")
logger.setLevel(logging.DEBUG)

# File handler — detailed log with timestamps
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(file_handler)

# Console handler — same output also goes to terminal
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}

# ── Manifest for Resume Support ──────────────────────────────────────────
MANIFEST_FILENAME = ".ingested.json"


def _manifest_path(volume_dir: Path) -> Path:
    """Path to the manifest file inside a volume directory."""
    return volume_dir / MANIFEST_FILENAME


def load_manifest(volume_dir: Path) -> set[str]:
    """Load the set of already-ingested filenames from the manifest."""
    mf = _manifest_path(volume_dir)
    if mf.exists():
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
            return set(data.get("ingested", []))
        except (json.JSONDecodeError, KeyError):
            return set()
    return set()


def save_manifest(volume_dir: Path, ingested: set[str]) -> None:
    """Save the set of ingested filenames to the manifest."""
    mf = _manifest_path(volume_dir)
    mf.write_text(
        json.dumps({"ingested": sorted(ingested)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def check_api() -> bool:
    """Check if the API is running and DB is connected."""
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=5)
        data = r.json()
        if data.get("db_connected"):
            logger.info("✅ API & Database connected")
            return True
        else:
            logger.warning("⚠️  API is up but database is not connected.")
            logger.warning("   Make sure PostgreSQL is running and ALLOYDB_* in .env is correct")
            return False
    except requests.ConnectionError:
        logger.error("❌ Cannot reach API at http://localhost:8000")
        logger.error("   Start it with: uvicorn backend.main:app --reload --port 8000")
        return False


def get_or_create_manga(title: str, language: str = "ja") -> int:
    """Get existing manga by title or create a new one. Returns manga_id."""
    # Check if manga already exists
    r = requests.get(f"{API_BASE}/api/manga")
    if r.status_code == 200:
        manga_list = r.json().get("manga", [])
        for m in manga_list:
            if m["title"] == title:
                logger.info(f"  📗 Found existing manga: {title} (id={m['id']})")
                return m["id"]

    # Create new manga
    r = requests.post(f"{API_BASE}/api/manga", json={"title": title, "language": language})
    if r.status_code == 200:
        manga_id = r.json()["manga_id"]
        logger.info(f"  📗 Created manga: {title} (id={manga_id})")
        return manga_id
    else:
        logger.error(f"  ❌ Failed to create manga: {title} — {r.text}")
        sys.exit(1)


def discover_volumes(source_dir: Path) -> dict[str, list[Path]]:
    """
    Discover images grouped by volume folder.
    
    Returns dict: { volume_folder_name: [sorted image paths] }
    
    If source_dir itself contains images (no sub-volumes), returns:
      { source_dir.name: [images] }
    """
    volumes: dict[str, list[Path]] = {}

    # Check if source_dir directly contains images
    direct_images = sorted(
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if direct_images:
        # Source dir IS a volume folder (e.g. downloads/进击的巨人/第01卷/)
        volumes[source_dir.name] = direct_images
    else:
        # Source dir contains sub-volume folders (e.g. downloads/进击的巨人/)
        for subdir in sorted(source_dir.iterdir()):
            if subdir.is_dir():
                imgs = sorted(
                    f for f in subdir.rglob("*")
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                )
                if imgs:
                    volumes[subdir.name] = imgs

    return volumes


def extract_page_number(filename: str) -> int | None:
    """
    Extract page number from filename.
    e.g. '001.webp' → 1, '042.png' → 42, 'cover.jpg' → None
    """
    stem = Path(filename).stem
    # Strip leading zeros and convert to int if the stem is purely digits
    if stem.isdigit():
        return int(stem)
    return None


def detect_manga_name(source: Path) -> str:
    """
    Detect the base manga name from folder structure.
    e.g. downloads/进击的巨人/第01卷 → "进击的巨人"
    e.g. downloads/进击的巨人 → "进击的巨人"
    """
    parts = source.parts
    if "downloads" in parts:
        dl_idx = parts.index("downloads")
        if dl_idx + 1 < len(parts):
            return parts[dl_idx + 1]
    return source.name


# Retry settings for transient errors at the HTTP layer
INGEST_MAX_RETRIES = 3
INGEST_RETRY_BASE_DELAY = 3.0  # seconds


async def ingest_image_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    image_path: Path,
    manga_id: int,
    manga_title: str,
    index: int,
    total: int,
    display_name: str,
    embedding_dim: int = 768,
) -> bool:
    """Ingest a single image through the API asynchronously with retry."""
    async with semaphore:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/webp"

        for attempt in range(INGEST_MAX_RETRIES + 1):
            try:
                data = aiohttp.FormData()
                data.add_field(
                    "file",
                    open(image_path, "rb"),
                    filename=image_path.name,
                    content_type=mime_type,
                )
                data.add_field("asset_type", "image")
                data.add_field("manga_id", str(manga_id))
                data.add_field("manga_title", manga_title)
                data.add_field("embedding_dim", str(embedding_dim))

                async with session.post(
                    f"{API_BASE}/api/ingest",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        retry_note = f" (retry {attempt})" if attempt > 0 else ""
                        logger.info(f"  [{index:>4}/{total}] {display_name} ... ✅ asset_id={result['asset_id']}{retry_note}")
                        return True
                    else:
                        text = await resp.text()
                        # Retry on 500/502/503/429 (server or rate-limit errors)
                        if resp.status in (429, 500, 502, 503) and attempt < INGEST_MAX_RETRIES:
                            delay = INGEST_RETRY_BASE_DELAY * (2 ** attempt)
                            logger.warning(
                                f"  [{index:>4}/{total}] {display_name} ... ⚠️  {resp.status}, retrying in {delay:.0f}s "
                                f"(attempt {attempt + 1}/{INGEST_MAX_RETRIES})"
                            )
                            await asyncio.sleep(delay)
                            continue
                        logger.error(f"  [{index:>4}/{total}] {display_name} ... ❌ {resp.status}: {text[:100]}")
                        return False

            except (aiohttp.ClientError, ConnectionResetError, OSError) as e:
                if attempt < INGEST_MAX_RETRIES:
                    delay = INGEST_RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"  [{index:>4}/{total}] {display_name} ... ⚠️  {e}, retrying in {delay:.0f}s "
                        f"(attempt {attempt + 1}/{INGEST_MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"  [{index:>4}/{total}] {display_name} ... ❌ {e} (exhausted retries)")
                    return False
            except Exception as e:
                logger.error(f"  [{index:>4}/{total}] {display_name} ... ❌ {e}")
                return False

        return False


async def ingest_volume_async(
    images: list[Path],
    manga_id: int,
    manga_title: str,
    volume_name: str,
    concurrency: int,
    volume_dir: Path | None = None,
    existing_manifest: set[str] | None = None,
    embedding_dim: int = 768,
) -> tuple[int, int, set[str]]:
    """
    Ingest all images in a volume concurrently.
    Returns (success, failed, set_of_successfully_ingested_filenames).
    
    If volume_dir is provided, the manifest is updated after each batch.
    existing_manifest tracks filenames already in the manifest (for accurate counting).

    Connection management:
    - Uses a TCPConnector with limit=concurrency to cap open TCP connections
      to the backend (prevents "Connection reset by peer" when the backend
      or its upstream services can't handle too many simultaneous connections).
    - Uses an asyncio.Semaphore(concurrency) for backpressure on task creation.
    - Adds a small stagger delay (0.15s) between launching tasks to avoid a
      thundering-herd of connections all opening at the same instant.
    """
    semaphore = asyncio.Semaphore(concurrency)
    total = len(images)
    newly_ingested: set[str] = set()
    manifest_lock = asyncio.Lock()

    # Stagger delay between launching concurrent tasks (seconds).
    # Prevents all N connections from hitting the server at the exact same
    # instant, which is the primary trigger for connection resets.
    STAGGER_DELAY = 0.15

    async def _ingest_and_track(session, img_path, idx):
        ok = await ingest_image_async(
            session, semaphore, img_path, manga_id, manga_title, idx, total,
            f"{volume_name}/{img_path.name}",
            embedding_dim=embedding_dim,
        )
        if ok:
            async with manifest_lock:
                newly_ingested.add(img_path.name)
                # Incrementally save manifest so progress survives crashes
                if volume_dir:
                    combined = (existing_manifest or set()) | newly_ingested
                    save_manifest(volume_dir, combined)
        return ok

    # TCPConnector limits the number of simultaneous TCP connections.
    # This is an additional safety net on top of the semaphore — even if
    # the semaphore allows N tasks to proceed, only `limit` connections
    # will actually be open at once.  keepalive reuses connections.
    connector = aiohttp.TCPConnector(
        limit=concurrency,
        limit_per_host=concurrency,
        keepalive_timeout=30,
        enable_cleanup_closed=True,
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i, img_path in enumerate(images, 1):
            tasks.append(asyncio.ensure_future(
                _ingest_and_track(session, img_path, i)
            ))
            # Stagger: wait a bit before launching the next task so
            # connections don't all open simultaneously
            if i < len(images):
                await asyncio.sleep(STAGGER_DELAY)

        results = await asyncio.gather(*tasks)

    success = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)
    return success, failed, newly_ingested


def main():
    parser = argparse.ArgumentParser(
        description="Async batch ingest manga images from downloads/ into Manga-Lingo"
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path to manga folder, e.g. 'downloads/进击的巨人/第01卷' or 'downloads/进击的巨人'",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override manga title (default: auto-detect with volume, e.g. '进击的巨人 第01卷')",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ja",
        help="Language code: 'ja' or 'ko' (default: ja)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of concurrent API requests (default: 3)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=768,
        choices=[768, 1536, 3072],
        help="Embedding dimension (default: 768). Higher = better quality, more cost/storage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be ingested without actually ingesting",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run — skip images already ingested (tracked via .ingested.json manifest)",
    )
    parser.add_argument(
        "--seed-manifest",
        type=str,
        nargs="?",
        const="ALL",
        metavar="FILE1,FILE2,...",
        help=(
            "Create .ingested.json manifest without running ingestion. "
            "Use alone to mark ALL images as ingested (e.g. after a mostly-successful run). "
            "Or pass comma-separated filenames to mark only those as ingested. "
            "Example: --seed-manifest 001.webp,002.webp,003.webp"
        ),
    )
    parser.add_argument(
        "--seed-manifest-exclude",
        type=str,
        metavar="FILE1,FILE2,...",
        help=(
            "Like --seed-manifest but marks all images EXCEPT the listed ones as ingested. "
            "Useful after a run with known failures. "
            "Example: --seed-manifest-exclude 184.webp,188.webp,191.webp"
        ),
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        logger.error(f"❌ Source path does not exist: {source}")
        sys.exit(1)

    # Detect base manga name
    manga_name = args.title or detect_manga_name(source)

    # Discover volumes and their images
    volumes = discover_volumes(source)

    if not volumes:
        logger.error("❌ No images found in the source directory.")
        sys.exit(1)

    total_images = sum(len(imgs) for imgs in volumes.values())

    logger.info("=" * 60)
    logger.info("📚 Manga-Lingo — Async Batch Ingest")
    logger.info("=" * 60)
    logger.info("")
    logger.info(f"  Source:       {source}")
    logger.info(f"  Manga:        {manga_name}")
    logger.info(f"  Volumes:      {len(volumes)}")
    logger.info(f"  Total images: {total_images}")
    logger.info(f"  Concurrency:  {args.concurrency}")
    logger.info(f"  Embed dim:    {args.embedding_dim}")
    logger.info(f"  Language:     {args.language}")
    logger.info(f"  Log file:     {log_filename}")
    logger.info("")

    for vol_name, imgs in volumes.items():
        title = f"{manga_name} {vol_name}" if vol_name != manga_name else manga_name
        logger.info(f"    📖 {title} — {len(imgs)} pages")

    logger.info("")

    if args.dry_run:
        logger.info("  🔍 DRY RUN — files that would be ingested:")
        for vol_name, imgs in volumes.items():
            title = f"{manga_name} {vol_name}" if vol_name != manga_name else manga_name
            logger.info(f"\n  --- {title} ---")
            for img in imgs:
                logger.info(f"     {img}")
        logger.info("")
        logger.info(f"  Total: {total_images} images across {len(volumes)} volume(s)")
        logger.info("  Run without --dry-run to actually ingest.")
        return

    # ── Seed Manifest (no ingestion) ────────────────────────────────
    if args.seed_manifest is not None or args.seed_manifest_exclude:
        for vol_name, imgs in volumes.items():
            volume_dir = imgs[0].parent if imgs else source
            all_names = {img.name for img in imgs}

            if args.seed_manifest_exclude:
                # Mark all EXCEPT the listed files
                exclude = set(args.seed_manifest_exclude.split(","))
                to_mark = all_names - exclude
                excluded_count = len(all_names & exclude)
                logger.info(f"  📝 {vol_name}: marking {len(to_mark)} as ingested, excluding {excluded_count}")
            elif args.seed_manifest == "ALL":
                # Mark ALL as ingested
                to_mark = all_names
                logger.info(f"  📝 {vol_name}: marking ALL {len(to_mark)} images as ingested")
            else:
                # Mark only specific files
                to_mark = set(args.seed_manifest.split(","))
                logger.info(f"  📝 {vol_name}: marking {len(to_mark)} specific files as ingested")

            # Merge with any existing manifest
            existing = load_manifest(volume_dir)
            combined = existing | to_mark
            save_manifest(volume_dir, combined)
            logger.info(f"  ✅ Saved manifest: {_manifest_path(volume_dir)}")
            logger.info(f"     Total tracked: {len(combined)} files")

        logger.info("")
        logger.info("Manifest seeded! Now run with --resume to ingest only the remaining files.")
        return

    # Check API
    logger.info("🔍 Checking API connection...")
    if not check_api():
        sys.exit(1)
    logger.info("")

    # Process each volume
    grand_success = 0
    grand_failed = 0
    grand_skipped = 0
    start_time = time.time()

    for vol_name, imgs in volumes.items():
        # Build title with volume: "进击的巨人 第01卷"
        if vol_name != manga_name:
            title = f"{manga_name} {vol_name}"
        else:
            title = manga_name

        # Resolve the actual volume directory (where images live / manifest goes)
        # imgs[0].parent gives us the directory containing the images
        volume_dir = imgs[0].parent if imgs else source

        # ── Resume: load manifest and filter out already-ingested ────
        already_ingested: set[str] = set()
        if args.resume:
            already_ingested = load_manifest(volume_dir)
            if already_ingested:
                before = len(imgs)
                imgs = [img for img in imgs if img.name not in already_ingested]
                skipped = before - len(imgs)
                grand_skipped += skipped
                logger.info(f"📖 Processing: {title} — {skipped} already ingested, {len(imgs)} remaining")
            else:
                logger.info(f"📖 Processing: {title} ({len(imgs)} pages)")
        else:
            logger.info(f"📖 Processing: {title} ({len(imgs)} pages)")

        if not imgs:
            logger.info(f"  ⏭️  All images already ingested — skipping volume")
            logger.info("")
            continue

        logger.info("")

        # Get or create manga entry with volume in title
        manga_id = get_or_create_manga(title, args.language)
        logger.info("")

        logger.info(f"🖼️  Ingesting {len(imgs)} images (concurrency={args.concurrency})...")
        logger.info("   (Each image calls Gemini Embedding 2)")
        logger.info("")

        success, failed, newly_done = asyncio.run(
            ingest_volume_async(
                imgs, manga_id, title, vol_name, args.concurrency,
                volume_dir=volume_dir,
                existing_manifest=already_ingested,
                embedding_dim=args.embedding_dim,
            )
        )
        grand_success += success
        grand_failed += failed

        logger.info("")
        logger.info(f"  ✅ {title}: {success} ingested, {failed} failed")
        if args.resume and grand_skipped:
            logger.info(f"     (previously ingested: {len(already_ingested)})")
        logger.info("")

    elapsed = time.time() - start_time
    actually_processed = grand_success + grand_failed
    logger.info("=" * 60)
    logger.info(f"✅ All done! {grand_success} ingested, {grand_failed} failed")
    if grand_skipped:
        logger.info(f"   ⏭️  {grand_skipped} skipped (already ingested from previous run)")
    logger.info(f"   Time: {elapsed:.1f}s ({elapsed/max(actually_processed,1):.1f}s per image)")
    if actually_processed > 0 and elapsed > 0:
        logger.info(f"   Throughput: {actually_processed/elapsed:.1f} images/sec")
    if grand_failed > 0:
        logger.info("")
        logger.info(f"   💡 Tip: Run again with --resume to retry the {grand_failed} failed images")
    logger.info("")
    logger.info("You can now search these images in the Streamlit UI!")
    logger.info(f"📄 Full log saved to: {log_filename}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
