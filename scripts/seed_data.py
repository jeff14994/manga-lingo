#!/usr/bin/env python3
"""
Manga-Lingo: Sample Data Seeder

Generates synthetic manga panels (images with Japanese/Korean text),
sample audio placeholders, and ingests them through the API.

Usage:
    # Make sure the backend is running first:
    #   uvicorn backend.main:app --reload --port 8000
    
    python scripts/seed_data.py
"""

import requests
import io
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow is required for generating sample images.")
    print("Install it: pip install Pillow")
    sys.exit(1)

API_BASE = "http://localhost:8000"

# ── Sample Manga Data ────────────────────────────────────────────────────

SAMPLE_MANGA = [
    {"title": "鬼滅の刃 (Demon Slayer)", "language": "ja"},
    {"title": "나 혼자만 레벨업 (Solo Leveling)", "language": "ko"},
    {"title": "ワンピース (One Piece)", "language": "ja"},
]

# Sample panels: each has dialogue text, translation, and a visual description
# used to generate a simple placeholder image
SAMPLE_PANELS = [
    # Demon Slayer panels
    {
        "manga_idx": 0,
        "dialogue_text": "全集中！水の呼吸！",
        "translation_en": "Total Concentration! Water Breathing!",
        "scene": "Blue water waves, samurai with sword",
        "colors": ("#1a237e", "#bbdefb"),
    },
    {
        "manga_idx": 0,
        "dialogue_text": "俺は鬼を倒す！",
        "translation_en": "I will defeat the demons!",
        "scene": "Angry warrior with glowing sword",
        "colors": ("#b71c1c", "#ffcdd2"),
    },
    {
        "manga_idx": 0,
        "dialogue_text": "禰豆子！大丈夫か？",
        "translation_en": "Nezuko! Are you okay?",
        "scene": "Worried boy reaching out",
        "colors": ("#2e7d32", "#c8e6c9"),
    },
    {
        "manga_idx": 0,
        "dialogue_text": "頑張れ！もう少しだ！",
        "translation_en": "Hang in there! Just a little more!",
        "scene": "Intense battle scene",
        "colors": ("#e65100", "#ffe0b2"),
    },
    # Solo Leveling panels
    {
        "manga_idx": 1,
        "dialogue_text": "나만 레벨업 한다",
        "translation_en": "Only I level up",
        "scene": "Dark silhouette with glowing eyes",
        "colors": ("#4a148c", "#e1bee7"),
    },
    {
        "manga_idx": 1,
        "dialogue_text": "일어나. 전투가 시작된다.",
        "translation_en": "Get up. The battle begins.",
        "scene": "Shadow warrior standing up",
        "colors": ("#1b5e20", "#a5d6a7"),
    },
    {
        "manga_idx": 1,
        "dialogue_text": "이 힘은... 뭐지?",
        "translation_en": "This power... what is it?",
        "scene": "Surprised expression with energy aura",
        "colors": ("#0d47a1", "#90caf9"),
    },
    # One Piece panels
    {
        "manga_idx": 2,
        "dialogue_text": "海賊王に俺はなる！",
        "translation_en": "I'm gonna be the Pirate King!",
        "scene": "Boy in straw hat pointing at sky",
        "colors": ("#f57f17", "#fff9c4"),
    },
    {
        "manga_idx": 2,
        "dialogue_text": "仲間を守るためなら、何だってする！",
        "translation_en": "I'll do anything to protect my friends!",
        "scene": "Determined face with fire background",
        "colors": ("#c62828", "#ef9a9a"),
    },
    {
        "manga_idx": 2,
        "dialogue_text": "ゴムゴムのぉ...ピストル！",
        "translation_en": "Gum Gum... Pistol!",
        "scene": "Stretching punch attack",
        "colors": ("#ad1457", "#f48fb1"),
    },
    {
        "manga_idx": 2,
        "dialogue_text": "冒険はまだ終わらない！",
        "translation_en": "The adventure isn't over yet!",
        "scene": "Ship sailing on the ocean",
        "colors": ("#00838f", "#80deea"),
    },
    {
        "manga_idx": 2,
        "dialogue_text": "夢を諦めるな！",
        "translation_en": "Don't give up on your dreams!",
        "scene": "Sunset with silhouette",
        "colors": ("#ff6f00", "#ffcc80"),
    },
]


def generate_manga_panel(
    dialogue: str,
    translation: str,
    scene: str,
    bg_color: str,
    text_color: str,
    width: int = 600,
    height: int = 800,
) -> bytes:
    """Generate a simple manga panel placeholder image."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Try to get a font, fall back to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_scene = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", 16)
    except (OSError, IOError):
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_scene = ImageFont.load_default()

    # Draw manga panel border
    draw.rectangle([10, 10, width - 10, height - 10], outline=text_color, width=3)

    # Draw "MANGA PANEL" label
    draw.text((20, 20), "📖 MANGA PANEL", fill=text_color, font=font_small)

    # Draw scene description in center
    draw.text(
        (width // 2, height // 3),
        f"[{scene}]",
        fill=text_color,
        font=font_scene,
        anchor="mm",
    )

    # Draw speech bubble area
    bubble_y = height // 2
    bubble_h = 180
    draw.rounded_rectangle(
        [40, bubble_y, width - 40, bubble_y + bubble_h],
        radius=20,
        fill="white",
        outline="black",
        width=2,
    )

    # Draw dialogue text
    draw.text(
        (width // 2, bubble_y + 40),
        dialogue,
        fill="black",
        font=font_large,
        anchor="mm",
    )

    # Draw translation
    draw.text(
        (width // 2, bubble_y + 90),
        translation,
        fill="#666666",
        font=font_small,
        anchor="mm",
    )

    # Draw bottom label
    draw.text(
        (width // 2, height - 40),
        "Sample Data — Manga-Lingo",
        fill=text_color,
        font=font_scene,
        anchor="mm",
    )

    # Convert to bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def check_api():
    """Check if the API is running."""
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=5)
        data = r.json()
        if data.get("db_connected"):
            print("✅ API & Database connected")
            return True
        else:
            print("⚠️  API is up but database is not connected.")
            print("    Make sure PostgreSQL is running and ALLOYDB_* in .env is correct")
            return False
    except requests.ConnectionError:
        print("❌ Cannot reach API at http://localhost:8000")
        print("   Start it with: uvicorn backend.main:app --reload --port 8000")
        return False


def create_manga_entries() -> list[int]:
    """Create manga entries and return their IDs."""
    manga_ids = []
    for manga in SAMPLE_MANGA:
        r = requests.post(f"{API_BASE}/api/manga", json=manga)
        if r.status_code == 200:
            data = r.json()
            manga_id = data["manga_id"]
            manga_ids.append(manga_id)
            print(f"  📗 Created manga: {manga['title']} (id={manga_id})")
        else:
            print(f"  ❌ Failed to create manga: {manga['title']} — {r.text}")
            manga_ids.append(None)
    return manga_ids


def ingest_panels(manga_ids: list[int]):
    """Generate and ingest sample manga panels."""
    for i, panel in enumerate(SAMPLE_PANELS):
        manga_id = manga_ids[panel["manga_idx"]]
        if manga_id is None:
            continue

        # Generate image
        img_bytes = generate_manga_panel(
            dialogue=panel["dialogue_text"],
            translation=panel["translation_en"],
            scene=panel["scene"],
            bg_color=panel["colors"][0],
            text_color=panel["colors"][1],
        )

        # Ingest via API
        files = {"file": (f"panel_{i:03d}.png", img_bytes, "image/png")}
        data = {
            "asset_type": "image",
            "manga_id": str(manga_id),
            "dialogue_text": panel["dialogue_text"],
            "translation_en": panel["translation_en"],
        }

        manga_title = SAMPLE_MANGA[panel["manga_idx"]]["title"]
        print(f"  🖼️  Ingesting panel {i+1}/{len(SAMPLE_PANELS)}: "
              f"{panel['dialogue_text'][:20]}... ({manga_title})")

        try:
            r = requests.post(f"{API_BASE}/api/ingest", files=files, data=data)
            if r.status_code == 200:
                result = r.json()
                print(f"     ✅ asset_id={result['asset_id']}, path={result['file_path']}")
            else:
                print(f"     ❌ Error: {r.text}")
        except Exception as e:
            print(f"     ❌ Exception: {e}")


def main():
    print("=" * 60)
    print("📚 Manga-Lingo — Sample Data Seeder")
    print("=" * 60)
    print()

    # Check API
    print("🔍 Checking API connection...")
    if not check_api():
        sys.exit(1)
    print()

    # Create manga entries
    print("📗 Creating manga entries...")
    manga_ids = create_manga_entries()
    print()

    # Ingest sample panels
    print(f"🖼️  Generating & ingesting {len(SAMPLE_PANELS)} sample panels...")
    print("   (Each panel calls Gemini Embedding 2 — this may take a moment)")
    print()
    ingest_panels(manga_ids)
    print()

    print("=" * 60)
    print("✅ Done! You can now search in the Streamlit UI.")
    print()
    print("Try these searches:")
    print("  • 'water breathing sword'")
    print("  • '海賊王' (Pirate King)")
    print("  • 'dark power level up'")
    print("  • '仲間を守る' (protect friends)")
    print("  • '전투' (battle)")
    print("=" * 60)


if __name__ == "__main__":
    main()
