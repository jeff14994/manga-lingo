"""
Japanese study service: dialogue lines (manga bubbles / video transcript),
word lookups, and SM-2 spaced-repetition review.
"""

import json
from datetime import datetime, timedelta, timezone

from backend.db import get_db
from backend.services import japanese

# ── Storing lines ────────────────────────────────────────────────────────


def save_lines(asset_id: int, source: str, lines: list[dict]) -> int:
    """
    Replace an asset's dialogue lines (and their words) with `lines`.

    Each line: text_ja, translation_en, grammar_note, and optionally bbox
    (manga) or t_start/t_end (video, absolute seconds).
    Also fills assets.dialogue_text / translation_en so search results show
    the dialogue. Returns the number of lines stored.
    """
    lines = [l for l in lines if l.get("text_ja", "").strip()]

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM dialogue_lines WHERE asset_id = %s", (asset_id,))

        for idx, line in enumerate(lines):
            text = line["text_ja"].strip()
            cursor.execute(
                """
                INSERT INTO dialogue_lines
                    (asset_id, line_index, source, text_ja, translation_en, grammar_note, bbox, t_start, t_end)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    asset_id, idx, source, text,
                    line.get("translation_en"), line.get("grammar_note"),
                    json.dumps(line["bbox"]) if line.get("bbox") else None,
                    line.get("t_start"), line.get("t_end"),
                ),
            )
            line_id = cursor.fetchone()[0]

            for w in japanese.study_words(japanese.tokenize(text)):
                cursor.execute(
                    """
                    INSERT INTO line_words (line_id, lemma, surface, reading, jlpt)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (line_id, w["lemma"], w["surface"], w["reading"], w["jlpt"]),
                )

        cursor.execute(
            "UPDATE assets SET dialogue_text = %s, translation_en = %s WHERE id = %s",
            (
                "\n".join(l["text_ja"].strip() for l in lines) or None,
                "\n".join(l.get("translation_en") or "" for l in lines).strip() or None,
                asset_id,
            ),
        )
        cursor.close()

    return len(lines)


# ── Reading ──────────────────────────────────────────────────────────────


def _rows(cursor) -> list[dict]:
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_asset_lines(asset_id: int) -> dict:
    """An asset plus its lines, each with furigana HTML and word breakdown."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT a.id AS asset_id, a.file_path, a.asset_type, a.page_number,
                   a.chunk_start, a.chunk_end, a.manga_id, m.title AS manga_title
            FROM assets a LEFT JOIN manga m ON a.manga_id = m.id
            WHERE a.id = %s
            """,
            (asset_id,),
        )
        assets = _rows(cursor)
        if not assets:
            cursor.close()
            return {}
        cursor.execute(
            """
            SELECT id AS line_id, line_index, source, text_ja, translation_en,
                   grammar_note, bbox, t_start, t_end
            FROM dialogue_lines WHERE asset_id = %s ORDER BY line_index
            """,
            (asset_id,),
        )
        lines = _rows(cursor)
        cursor.close()

    for line in lines:
        tokens = japanese.tokenize(line["text_ja"])
        line["furigana_html"] = japanese.furigana_html(line["text_ja"], tokens)
        line["words"] = japanese.study_words(tokens)

    return {**assets[0], "lines": lines}


def list_pages(manga_id: int) -> list[dict]:
    """
    Pages (image assets) or video chunks of a title in reading/playback
    order, with how many dialogue lines each has.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT a.id AS asset_id, a.asset_type, a.page_number, a.chunk_start,
                   a.chunk_end, a.file_path, COUNT(l.id) AS line_count
            FROM assets a LEFT JOIN dialogue_lines l ON l.asset_id = a.id
            WHERE a.manga_id = %s AND a.asset_type IN ('image', 'video')
            GROUP BY a.id
            ORDER BY a.page_number NULLS LAST, a.chunk_start NULLS LAST, a.id
            """,
            (manga_id,),
        )
        rows = _rows(cursor)
        cursor.close()
    return rows


def word_occurrences(lemma: str, source: str | None = None, limit: int = 20) -> list[dict]:
    """
    Lines that use a word (by dictionary form). For video lines, offset is
    the playback position inside the chunk file (t_start - chunk_start).
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT ON (l.id)
                   l.id AS line_id, l.source, l.text_ja, l.translation_en,
                   l.t_start, l.t_end, a.id AS asset_id, a.file_path,
                   a.asset_type, a.page_number, a.chunk_start,
                   m.title AS manga_title
            FROM line_words w
            JOIN dialogue_lines l ON l.id = w.line_id
            JOIN assets a ON a.id = l.asset_id
            LEFT JOIN manga m ON m.id = a.manga_id
            WHERE w.lemma = %s AND (%s::text IS NULL OR l.source = %s)
            ORDER BY l.id
            LIMIT %s
            """,
            (lemma, source, source, limit),
        )
        rows = _rows(cursor)
        cursor.close()

    for row in rows:
        if row["t_start"] is not None and row["chunk_start"] is not None:
            row["offset"] = max(0.0, row["t_start"] - row["chunk_start"])
    return rows


# ── Review (SM-2) ────────────────────────────────────────────────────────

GRADES = {"again": 0, "hard": 3, "good": 4, "easy": 5}


def add_review_item(lemma: str, example_line_id: int | None = None) -> dict:
    """Add a word to review (no-op if it's already there)."""
    word = next(iter(japanese.study_words(japanese.tokenize(lemma))), None)
    reading = word["reading"] if word and word["lemma"] == lemma else None
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO review_items (lemma, reading, meaning, jlpt, example_line_id)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (lemma) DO UPDATE
                SET example_line_id = COALESCE(review_items.example_line_id, EXCLUDED.example_line_id)
            RETURNING id, lemma, due_at
            """,
            (lemma, reading, japanese.meaning(lemma), japanese.jlpt_level(lemma, reading), example_line_id),
        )
        row = _rows(cursor)[0]
        cursor.close()
    return row


def due_items(limit: int = 20) -> list[dict]:
    """Review cards due now, with their example line and source asset."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT r.id, r.lemma, r.reading, r.meaning, r.jlpt, r.reps,
                   r.interval_days, r.due_at,
                   l.text_ja, l.translation_en, l.source,
                   a.id AS asset_id, a.file_path, a.asset_type
            FROM review_items r
            LEFT JOIN dialogue_lines l ON l.id = r.example_line_id
            LEFT JOIN assets a ON a.id = l.asset_id
            WHERE r.due_at <= NOW()
            ORDER BY r.due_at
            LIMIT %s
            """,
            (limit,),
        )
        rows = _rows(cursor)
        cursor.close()

    for row in rows:
        if row["text_ja"]:
            row["furigana_html"] = japanese.furigana_html(row["text_ja"])
    return rows


def grade_item(item_id: int, grade: str) -> dict:
    """
    Apply an SM-2 update. 'again' resets the card to be seen again in
    10 minutes; other grades grow the interval by the ease factor.
    """
    q = GRADES[grade]
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ease, interval_days, reps, lapses FROM review_items WHERE id = %s FOR UPDATE",
            (item_id,),
        )
        row = cursor.fetchone()
        if row is None:
            cursor.close()
            raise KeyError(item_id)
        ease, interval, reps, lapses = row

        now = datetime.now(timezone.utc)
        if q < 3:
            reps, lapses, interval = 0, lapses + 1, 0.0
            due = now + timedelta(minutes=10)
        else:
            reps += 1
            if reps == 1:
                interval = 1.0
            elif reps == 2:
                interval = 6.0
            else:
                interval = interval * ease
            if grade == "hard":
                interval = max(1.0, interval * 0.6)
            elif grade == "easy":
                interval *= 1.3
            due = now + timedelta(days=interval)
        ease = max(1.3, ease + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))

        cursor.execute(
            """
            UPDATE review_items
            SET ease = %s, interval_days = %s, reps = %s, lapses = %s, due_at = %s
            WHERE id = %s
            """,
            (ease, interval, reps, lapses, due, item_id),
        )
        cursor.close()

    return {"id": item_id, "interval_days": interval, "due_at": due.isoformat(), "ease": ease}
