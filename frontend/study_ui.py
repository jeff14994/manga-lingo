"""
Manga-Lingo — Japanese study UI (Study + Review tabs).

Study:  read a page (or anime chunk) bubble by bubble with furigana,
        word meanings, JLPT levels and grammar notes; add words to review;
        hear a word in the anime.
Review: spaced-repetition cards; the answer shows the source panel and the
        anime moment where the word is spoken.
"""

import html
import urllib.parse
from typing import Callable

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# JLPT badge colours; anything harder than N4 is highlighted as "new" for an N5–N4 reader.
JLPT_STYLE = {
    5: ("N5", "#2e7d32"),
    4: ("N4", "#1565c0"),
    3: ("N3", "#ef6c00"),
    2: ("N2", "#c62828"),
    1: ("N1", "#6a1b9a"),
    None: ("—", "#616161"),
}

_CSS = """
<style>
.ml-line { font-size: 1.35rem; line-height: 2.3rem; margin: 0; }
.ml-line rt { font-size: 0.55em; color: #9e9e9e; }
.ml-badge { display: inline-block; padding: 0 0.45em; border-radius: 0.4em;
            color: white; font-size: 0.8rem; font-weight: 600; }
.ml-new { background: rgba(239,108,0,0.15); border-radius: 0.3em; padding: 0 0.2em; }
.ml-muted { color: #9e9e9e; font-size: 0.9rem; }
.ml-note { font-size: 0.9rem; margin-top: 0.15rem; }
.ml-chip { display: inline-block; padding: 0.05em 0.6em; border-radius: 1em; font-size: 0.8rem;
           border: 1px solid #616161; color: #bdbdbd; }
</style>
"""


def _jlpt_badge(level: int | None) -> str:
    label, color = JLPT_STYLE.get(level, JLPT_STYLE[None])
    return f'<span class="ml-badge" style="background:{color}">{label}</span>'


def _is_new_for_learner(level: int | None) -> bool:
    """Words above N4 (or unlisted) are 'new' for an N5–N4 learner."""
    return level is None or level <= 3


def _get(path: str, **params) -> dict:
    r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _post(path: str, payload: dict) -> dict:
    r = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def _quote(lemma: str) -> str:
    return urllib.parse.quote(lemma, safe="")


def _asset_label(page: dict) -> str:
    if page["asset_type"] == "video" and page.get("chunk_start") is not None:
        s = int(page["chunk_start"])
        label = f"{s // 60}:{s % 60:02d}"
    elif page.get("page_number") is not None:
        label = f"p.{page['page_number']}"
    else:
        label = f"#{page['asset_id']}"
    return f"{label} ({page['line_count']} lines)" if page["line_count"] else f"{label} (—)"


# ── Shared pieces ────────────────────────────────────────────────────────


def _play_word_in_anime(lemma: str, file_url_for: Callable[[dict], str], key: str):
    """Show the first anime moment where the word is spoken."""
    occ = _get(f"/api/words/{_quote(lemma)}/occurrences", source="video", limit=5)["occurrences"]
    if not occ:
        st.caption(f"🎬 「{lemma}」 isn't spoken in the transcribed episodes yet.")
        return
    choice = 0
    if len(occ) > 1:
        choice = st.radio(
            "Moments",
            range(len(occ)),
            format_func=lambda i: f"{int(occ[i]['t_start']) // 60}:{int(occ[i]['t_start']) % 60:02d} {occ[i]['text_ja']}",
            key=f"moment_{key}",
            horizontal=False,
        )
    o = occ[choice]
    st.video(file_url_for(o), start_time=int(o.get("offset", 0)))
    st.markdown(f"**{o['text_ja']}**")
    if o.get("translation_en"):
        st.caption(o["translation_en"])


def _render_word_row(word: dict, line_id: int, file_url_for: Callable[[dict], str], key: str):
    reading = f"（{word['reading']}）" if word.get("reading") and word["reading"] != word["lemma"] else ""
    lemma_html = f"<b>{word['lemma']}</b>{reading}"
    if _is_new_for_learner(word.get("jlpt")):
        lemma_html = f'<span class="ml-new">{lemma_html}</span>'

    c1, c2, c3 = st.columns([5, 1, 1])
    with c1:
        st.markdown(
            f"{_jlpt_badge(word.get('jlpt'))} {lemma_html} — {word.get('meaning') or '<i>no dictionary entry</i>'}",
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("➕", key=f"add_{key}", help="Add to review"):
            _post("/api/review/items", {"lemma": word["lemma"], "example_line_id": line_id})
            st.toast(f"Added 「{word['lemma']}」 to review")
    with c3:
        hear_key = f"hear_{key}"
        if st.button("🎬", key=f"btn_{hear_key}", help="Hear it in the anime"):
            st.session_state[hear_key] = not st.session_state.get(hear_key, False)
    if st.session_state.get(f"hear_{key}"):
        _play_word_in_anime(word["lemma"], file_url_for, key)


# ── Study tab ────────────────────────────────────────────────────────────


def render_study_tab(file_url_for: Callable[[dict], str]):
    st.markdown(_CSS, unsafe_allow_html=True)

    try:
        manga = _get("/api/manga")["manga"]
    except Exception as e:
        st.error(f"Cannot reach API: {e}")
        return
    if not manga:
        st.info("No titles yet. Ingest a manga or episode, then run "
                "`python scripts/extract_dialogue.py`.")
        return

    # Jump straight to an asset picked from search results ("📖 Study" button)
    jump = st.session_state.pop("study_jump", None)

    titles = {m["id"]: m["title"] for m in manga}
    languages = {m["id"]: m.get("language") for m in manga}
    ids = list(titles)
    # Widget values live in session_state once keyed, so set state (not index)
    if jump and jump.get("manga_id") in ids:
        st.session_state["study_manga"] = jump["manga_id"]
    if st.session_state.get("study_manga") not in ids:
        st.session_state.pop("study_manga", None)

    # ── Toolbar: title · page · ◀ ▶ ──────────────────────────────────────
    c_title, c_page, c_prev, c_next = st.columns([3, 2, 0.45, 0.45], vertical_alignment="bottom")
    with c_title:
        manga_id = st.selectbox("Title", ids, format_func=titles.get, key="study_manga")

    pages = _get(f"/api/manga/{manga_id}/pages")["pages"]
    if not pages:
        st.info("This title has no pages or clips.")
        return
    page_ids = [p["asset_id"] for p in pages]
    labels = {p["asset_id"]: _asset_label(p) for p in pages}
    if jump and jump.get("asset_id") in page_ids:
        st.session_state["study_asset"] = jump["asset_id"]
    if st.session_state.get("study_asset") not in page_ids:
        # Stale selection from another title
        st.session_state.pop("study_asset", None)

    def _step(delta: int):
        cur = st.session_state.get("study_asset", page_ids[0])
        i = page_ids.index(cur) if cur in page_ids else 0
        st.session_state["study_asset"] = page_ids[max(0, min(len(page_ids) - 1, i + delta))]

    with c_page:
        asset_id = st.selectbox("Page / scene", page_ids, format_func=labels.get, key="study_asset")
    idx = page_ids.index(asset_id)
    with c_prev:
        st.button("◀", on_click=_step, args=(-1,), disabled=idx == 0,
                  use_container_width=True, help="Previous page")
    with c_next:
        st.button("▶", on_click=_step, args=(1,), disabled=idx == len(page_ids) - 1,
                  use_container_width=True, help="Next page")

    data = _get(f"/api/assets/{asset_id}/lines")

    if data["asset_type"] == "video":
        col_video, col_lines = st.columns([1.3, 1], gap="medium")
        with col_video:
            _seekable_video(data, file_url_for, key=f"clip{asset_id}")
        with col_lines:
            _render_transcript(data, data["lines"], file_url_for, key_prefix=f"clip{asset_id}")
    else:
        # Manga page and its anime scene side by side
        col_page, col_scene = st.columns([1, 1.6], gap="medium")
        with col_page:
            st.image(file_url_for(data), use_container_width=True)
            if languages[manga_id] != "ja":
                st.markdown(
                    '<span class="ml-chip">Not a Japanese edition</span> '
                    '<span class="ml-muted">hear this moment in Japanese →</span>',
                    unsafe_allow_html=True,
                )
            elif not data["lines"]:
                st.caption("No dialogue extracted yet. Run "
                           "`python scripts/extract_dialogue.py --title \"%s\"`." % titles[manga_id])
            for line in data["lines"]:
                _render_line(line, file_url_for, key_prefix="page")
        with col_scene:
            _render_anime_scene(asset_id, file_url_for)

    st.divider()
    _render_word_search(file_url_for)


def _seek_key(key: str) -> str:
    return f"seek_{key}"


def _seekable_video(clip: dict, file_url_for: Callable[[dict], str], key: str):
    """Video that jumps to (and plays from) the line chosen with ▶."""
    start = st.session_state.get(_seek_key(key))
    st.video(file_url_for(clip), start_time=int(start or 0), autoplay=start is not None)


def _render_transcript(clip: dict, lines: list[dict], file_url_for: Callable[[dict], str], key_prefix: str):
    """Scrollable transcript; each line has ▶ to play the clip from that line."""
    if not lines:
        st.caption("No spoken Japanese transcribed in this clip.")
        return
    with st.container(height=330, border=False):
        for line in lines:
            offset = None
            if line.get("t_start") is not None and clip.get("chunk_start") is not None:
                offset = max(0.0, line["t_start"] - clip["chunk_start"])
            _render_line(line, file_url_for, key_prefix=key_prefix,
                         seek=(key_prefix, offset) if offset is not None else None)


def _render_line(line: dict, file_url_for: Callable[[dict], str], key_prefix: str,
                 seek: tuple[str, float] | None = None):
    """
    One dialogue line: furigana, translation, grammar note, word list.
    `seek` = (video key, offset) adds a ▶ m:ss button that plays from this line.
    """
    with st.container(border=True):
        if seek is not None:
            c_play, c_text = st.columns([0.22, 1], vertical_alignment="center")
            with c_play:
                s = int(line["t_start"])
                if st.button(f"▶ {s // 60}:{s % 60:02d}", key=f"play_{key_prefix}_{line['line_id']}",
                             help="Play from this line"):
                    st.session_state[_seek_key(seek[0])] = seek[1]
                    st.rerun()
        else:
            c_text = st.container()
        with c_text:
            st.markdown(f'<div class="ml-line">{line["furigana_html"]}</div>', unsafe_allow_html=True)
            meta = []
            if line.get("translation_en"):
                meta.append(f'<div class="ml-muted">{html.escape(line["translation_en"])}</div>')
            if line.get("grammar_note"):
                meta.append(f'<div class="ml-note">📝 {html.escape(line["grammar_note"])}</div>')
            if meta:
                st.markdown("".join(meta), unsafe_allow_html=True)
        if line["words"]:
            with st.expander(f"Words ({len(line['words'])})"):
                for w in line["words"]:
                    _render_word_row(
                        w, line["line_id"], file_url_for,
                        key=f"{key_prefix}_{line['line_id']}_{w['lemma']}",
                    )


def _render_anime_scene(asset_id: int, file_url_for: Callable[[dict], str]):
    """
    Manga page → matching anime scene via the page's stored embedding
    (image→video search), with the scene's Japanese transcript.
    """
    dim = st.session_state.get("embedding_dim", 768)
    try:
        scenes = _get(f"/api/assets/{asset_id}/scenes", limit=3, embedding_dim=dim)["scenes"]
    except Exception as e:
        st.error(f"Scene search failed: {e}")
        return
    if not scenes:
        st.info(f"No anime clips with {dim}d embeddings yet.")
        return

    def _time(s: dict) -> str:
        t = int(s.get("chunk_start") or 0)
        return f"{t // 60}:{t % 60:02d}"

    marks = "①②③④⑤"
    c_head, c_pick = st.columns([1, 1.3], vertical_alignment="center")
    with c_head:
        st.markdown(
            "**🎬 This scene in the anime**",
            help=(f"Ranked by comparing this page's {dim}d Gemini embedding with every anime clip "
                  "(no shared words needed). Similarity: "
                  + " · ".join(f"{marks[i]} {1 - s['distance']:.0%}" for i, s in enumerate(scenes))),
        )
    with c_pick:
        choice = st.segmented_control(
            "Match", list(range(len(scenes))), default=0, key=f"scene_choice_{asset_id}",
            format_func=lambda i: f"{marks[i]} {_time(scenes[i])}", label_visibility="collapsed",
        )
    scene = scenes[choice if choice is not None else 0]
    # Seek state is per clip, so switching matches starts the new clip from the top
    clip_key = f"scene{asset_id}_{scene['asset_id']}"
    _seekable_video(scene, file_url_for, key=clip_key)
    _render_transcript(scene, scene["lines"], file_url_for, key_prefix=clip_key)


def _render_word_search(file_url_for: Callable[[dict], str]):
    st.subheader("🔎 Find a word in manga & anime")
    lemma = st.text_input("Dictionary form, e.g. 食べる, 壁, 巨人", key="word_search")
    if not lemma:
        return
    occ = _get(f"/api/words/{_quote(lemma.strip())}/occurrences", limit=12)["occurrences"]
    if not occ:
        st.info(f"「{lemma}」 not found in any extracted dialogue.")
        return
    cols = st.columns(3)
    for i, o in enumerate(occ):
        with cols[i % 3]:
            if o["source"] == "video":
                st.video(file_url_for(o), start_time=int(o.get("offset", 0)))
            else:
                st.image(file_url_for(o), use_container_width=True)
            st.markdown(f"**{o['text_ja']}**")
            if o.get("translation_en"):
                st.caption(o["translation_en"])


# ── Review tab ───────────────────────────────────────────────────────────


def render_review_tab(file_url_for: Callable[[dict], str]):
    st.markdown(_CSS, unsafe_allow_html=True)

    try:
        due = _get("/api/review/due", limit=50)["items"]
    except Exception as e:
        st.error(f"Cannot reach API: {e}")
        return
    if not due:
        st.success("🎉 Nothing due. Add words with ➕ in the 📖 Study tab.")
        return

    card = due[0]
    st.caption(f"{len(due)} due")
    revealed = st.session_state.get("review_revealed") == card["id"]

    with st.container(border=True):
        # Front: the word in context, without furigana
        sentence = card.get("text_ja") or card["lemma"]
        st.markdown(f'<div class="ml-line">{sentence}</div>', unsafe_allow_html=True)
        st.markdown(f"What does **{card['lemma']}** mean, and how is it read?")

        if not revealed:
            if st.button("Show answer", key=f"reveal_{card['id']}", type="primary"):
                st.session_state["review_revealed"] = card["id"]
                st.rerun()
            return

        # Back
        st.markdown(
            f"{_jlpt_badge(card.get('jlpt'))} <b>{card['lemma']}</b>（{card.get('reading') or '?'}） — "
            f"{card.get('meaning') or ''}",
            unsafe_allow_html=True,
        )
        if card.get("furigana_html"):
            st.markdown(f'<div class="ml-line">{card["furigana_html"]}</div>', unsafe_allow_html=True)
        if card.get("translation_en"):
            st.caption(f"🇺🇸 {card['translation_en']}")

        c_src, c_anime = st.columns(2)
        with c_src:
            if card.get("file_path"):
                st.caption("Where you met it")
                if card.get("asset_type") == "video":
                    st.video(file_url_for(card))
                else:
                    st.image(file_url_for(card), use_container_width=True)
        with c_anime:
            st.caption("🎬 Spoken in the anime")
            _play_word_in_anime(card["lemma"], file_url_for, key=f"review_{card['id']}")

        cols = st.columns(4)
        for col, (grade, label) in zip(cols, [("again", "🔁 Again"), ("hard", "😓 Hard"), ("good", "🙂 Good"), ("easy", "😎 Easy")]):
            with col:
                if st.button(label, key=f"grade_{grade}_{card['id']}", use_container_width=True):
                    _post(f"/api/review/{card['id']}/grade", {"grade": grade})
                    st.session_state.pop("review_revealed", None)
                    st.rerun()
