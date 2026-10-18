"""
Dialogue extraction with Gemini: speech bubbles from manga pages and
timestamped transcripts from video chunks. Readings/furigana are NOT taken
from the model — those come from backend.services.japanese.
"""

from backend.services.embedding import _convert_image_to_jpeg
from backend.services.gemini import generate_json

_LEARNER_NOTE = (
    "The reader is a Japanese learner at JLPT N5–N4. "
    "translation_en: natural English. "
    "grammar_note: ONE short line explaining the most useful grammar point or "
    "expression in the line for that level (e.g. '〜てしまう: did something "
    "completely / regrettably'); empty string if nothing notable."
)

MANGA_PROMPT = f"""Extract every line of Japanese dialogue and narration from this manga page.
- One item per speech bubble or caption box, in Japanese manga reading order:
  panels right-to-left, top-to-bottom; within a panel, right-to-left.
- Copy the Japanese text exactly as printed (keep kanji/kana as written, join
  vertical lines of one bubble into one string, drop furigana printed beside kanji).
- Skip sound effects (onomatopoeia drawn as art), page numbers and credits.
- bbox: the bubble's box as [ymin, xmin, ymax, xmax] scaled 0–1000.
{_LEARNER_NOTE}
Return an empty list if the page has no Japanese text."""

VIDEO_PROMPT = f"""Transcribe the spoken Japanese dialogue in this anime clip.
- One item per spoken line, in order.
- start/end: seconds from the beginning of THIS clip (decimals allowed).
- text_ja: what is actually said, in natural Japanese orthography (kanji + kana).
- Skip song lyrics, background music and non-speech sounds. Ignore any
  burned-in subtitles; transcribe the audio.
{_LEARNER_NOTE}
Return an empty list if nobody speaks."""

_LINE_PROPS = {
    "text_ja": {"type": "STRING"},
    "translation_en": {"type": "STRING"},
    "grammar_note": {"type": "STRING"},
}

MANGA_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            **_LINE_PROPS,
            "bbox": {"type": "ARRAY", "items": {"type": "INTEGER"}},
        },
        "required": ["text_ja", "translation_en", "grammar_note"],
    },
}

VIDEO_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            **_LINE_PROPS,
            "start": {"type": "NUMBER"},
            "end": {"type": "NUMBER"},
        },
        "required": ["text_ja", "translation_en", "grammar_note", "start", "end"],
    },
}


def extract_manga_lines(image_bytes: bytes) -> list[dict]:
    """Speech bubbles on a manga page, in reading order."""
    data, mime = _convert_image_to_jpeg(image_bytes)  # Gemini doesn't take WebP inline
    lines = generate_json(MANGA_PROMPT, [(data, mime)], MANGA_SCHEMA)
    for line in lines:
        if len(line.get("bbox") or []) != 4:
            line["bbox"] = None
    return lines


def transcribe_video_chunk(video_bytes: bytes, chunk_start: float = 0.0) -> list[dict]:
    """Spoken lines in a video chunk, with absolute t_start/t_end."""
    lines = generate_json(VIDEO_PROMPT, [(video_bytes, "video/mp4")], VIDEO_SCHEMA)
    for line in lines:
        line["t_start"] = chunk_start + float(line.pop("start", 0) or 0)
        line["t_end"] = chunk_start + float(line.pop("end", 0) or 0)
    return lines
