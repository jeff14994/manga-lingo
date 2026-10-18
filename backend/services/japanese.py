"""
Local Japanese text analysis for reading practice (no LLM involved, so
readings are deterministic):

- Tokenization + readings: fugashi with UniDic (unidic-lite)
- Furigana: <ruby> HTML over kanji only, with okurigana trimmed
- Glosses: JMdict via jamdict
- JLPT level: backend/data/jlpt_vocab.csv (Tanos lists, CC BY)

Attribution: JMdict is © EDRDG (CC BY-SA 4.0); JLPT lists by Jonathan
Waller (tanos.co.uk, CC BY).
"""

import csv
import html
import re
import threading
from functools import lru_cache
from pathlib import Path

JLPT_CSV = Path(__file__).resolve().parent.parent / "data" / "jlpt_vocab.csv"

# Parts of speech worth studying / adding to review.
CONTENT_POS = {"名詞", "代名詞", "動詞", "形容詞", "形状詞", "副詞", "連体詞", "感動詞"}

_KANJI_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿々〆ヶ]")

_tagger = None
_tagger_lock = threading.Lock()
# jamdict holds a SQLite connection, which can only be used by the thread
# that created it — keep one instance per thread.
_jam_local = threading.local()


def _get_tagger():
    global _tagger
    if _tagger is None:
        import fugashi
        _tagger = fugashi.Tagger()
    return _tagger


def kata_to_hira(text: str) -> str:
    """Convert katakana to hiragana (leaves everything else unchanged)."""
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def has_kanji(text: str) -> bool:
    return bool(_KANJI_RE.search(text))


@lru_cache(maxsize=1)
def _jlpt_table() -> tuple[dict, dict, dict]:
    """Return ({(expression, reading): level}, {expression: easiest level}, {reading: [levels]})."""
    by_pair, by_expr, by_reading = {}, {}, {}
    with open(JLPT_CSV, encoding="utf-8") as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in rows:
            level = int(row["level"])
            by_pair[(row["expression"], row["reading"])] = level
            by_expr[row["expression"]] = max(level, by_expr.get(row["expression"], 0))
            by_reading.setdefault(row["reading"], []).append(level)
    return by_pair, by_expr, by_reading


def jlpt_level(base: str, reading: str | None = None) -> int | None:
    """JLPT level 5 (easiest) … 1 (hardest), or None if not on the lists."""
    by_pair, by_expr, by_reading = _jlpt_table()
    # The lists sometimes use a different spelling: ご飯/御飯, お茶/御茶
    for expr in (base, re.sub(r"^[ごお]", "御", base)):
        if reading and (expr, reading) in by_pair:
            return by_pair[(expr, reading)]
        if expr in by_expr:
            return by_expr[expr]
    # Kana-only spelling of a word the lists write in kanji (ください/下さい).
    # Only when the reading is unambiguous, so homophones (かみ: 紙/髪/神) don't match.
    if not has_kanji(base) and len(by_reading.get(kata_to_hira(base), [])) == 1:
        return by_reading[kata_to_hira(base)][0]
    return None


@lru_cache(maxsize=4096)
def meaning(base: str, max_senses: int = 2) -> str | None:
    """Short English gloss from JMdict, e.g. 'to eat; to live on'."""
    jam = getattr(_jam_local, "jam", None)
    if jam is None:
        from jamdict import Jamdict
        jam = _jam_local.jam = Jamdict()
    result = jam.lookup(base, strict_lookup=True)
    if not result.entries:
        return None
    senses = result.entries[0].senses[:max_senses]
    return "; ".join(", ".join(g.text for g in s.gloss[:3]) for s in senses) or None


def tokenize(text: str) -> list[dict]:
    """
    Split Japanese text into tokens.

    Each token: surface, base (dictionary form as written), reading
    (hiragana of the surface), base_reading, pos, is_content.
    """
    with _tagger_lock:
        words = list(_get_tagger()(text))

    tokens = []
    for w in words:
        f = w.feature
        kana = getattr(f, "kana", None)
        base = getattr(f, "orthBase", None) or w.surface
        base_kana = getattr(f, "kanaBase", None)
        tokens.append({
            "surface": w.surface,
            "base": base,
            "reading": kata_to_hira(kana) if kana and kana != "*" else None,
            "base_reading": kata_to_hira(base_kana) if base_kana and base_kana != "*" else None,
            "pos": f.pos1,
            "is_content": f.pos1 in CONTENT_POS and bool(kana),
        })
    return tokens


def _ruby(surface: str, reading: str | None) -> str:
    """Wrap the kanji part of one token in <ruby>, keeping okurigana outside."""
    if not reading or not has_kanji(surface) or kata_to_hira(surface) == reading:
        return html.escape(surface)

    # Align kanji runs to the reading using the kana between them as anchors:
    # 取り扱い + とりあつかい → 取(と) り 扱(あつか) い
    segments = re.findall(rf"{_KANJI_RE.pattern}+|(?:(?!{_KANJI_RE.pattern}).)+", surface)
    pattern = "".join(
        "(.+?)" if has_kanji(seg) else re.escape(kata_to_hira(seg)) for seg in segments
    )
    m = re.fullmatch(pattern, reading)
    if m:
        groups = iter(m.groups())
        return "".join(
            f"<ruby>{html.escape(seg)}<rt>{html.escape(next(groups))}</rt></ruby>"
            if has_kanji(seg) else html.escape(seg)
            for seg in segments
        )

    # Fallback: trim kana shared at the start/end (お茶 → お + 茶, 食べ → 食 + べ)
    head = tail = ""
    s, r = surface, reading
    while s and r and not has_kanji(s[0]) and kata_to_hira(s[0]) == r[0]:
        head += s[0]
        s, r = s[1:], r[1:]
    while s and r and not has_kanji(s[-1]) and kata_to_hira(s[-1]) == r[-1]:
        tail = s[-1] + tail
        s, r = s[:-1], r[:-1]
    if not s or not r:
        return html.escape(surface)

    return f"{html.escape(head)}<ruby>{html.escape(s)}<rt>{html.escape(r)}</rt></ruby>{html.escape(tail)}"


def furigana_html(text: str, tokens: list[dict] | None = None) -> str:
    """Render text as HTML with <ruby> furigana over kanji."""
    tokens = tokens if tokens is not None else tokenize(text)
    return "".join(_ruby(t["surface"], t["reading"]) for t in tokens)


def study_words(tokens: list[dict]) -> list[dict]:
    """
    Content words from a token list, de-duplicated by base form, with
    reading, JLPT level and English gloss.
    """
    seen = set()
    words = []
    for t in tokens:
        if not t["is_content"] or t["base"] in seen:
            continue
        seen.add(t["base"])
        words.append({
            "lemma": t["base"],
            "surface": t["surface"],
            "reading": t["base_reading"],
            "jlpt": jlpt_level(t["base"], t["base_reading"]),
            "meaning": meaning(t["base"]),
        })
    return words
