"""
Manga-Lingo MVP — Streamlit Frontend
"""

import urllib.parse

import streamlit as st
import requests
from pathlib import Path

from study_ui import render_review_tab, render_study_tab

API_BASE = "http://localhost:8000"


def file_url_for(item: dict) -> str:
    """
    Build the URL to serve a file for a search result item.

    Prefers /api/serve/{file_path} which serves files directly by path
    without needing a DB lookup. This avoids DB connection pool exhaustion
    when the browser loads many images concurrently.

    Falls back to /api/file/{asset_id} (DB lookup) only when file_path
    is not available.
    """
    file_path = item.get("file_path", "")
    if file_path:
        # Use /api/serve/ endpoint — no DB connection needed, handles
        # Unicode (CJK) paths via FastAPI path params.
        # URL-encode each path segment to handle non-ASCII characters.
        encoded_path = "/".join(
            urllib.parse.quote(segment, safe="")
            for segment in file_path.split("/")
            if segment
        )
        return f"{API_BASE}/api/serve/{encoded_path}"

    # Fallback: use asset_id-based endpoint (requires DB lookup)
    asset_id = item.get("asset_id")
    if asset_id:
        return f"{API_BASE}/api/file/{asset_id}"

    return ""

st.set_page_config(page_title="Manga-Lingo", page_icon="📚", layout="wide")

# Initialize selected results list (max 2 items)
if "selected_results" not in st.session_state:
    st.session_state["selected_results"] = []

# Migrate old single-item state if present
if "selected_result" in st.session_state and st.session_state["selected_result"]:
    st.session_state["selected_results"] = [st.session_state.pop("selected_result")]


def add_selected(item: dict):
    """Add an item to the selected results (max 2). Replaces oldest if full."""
    sel = st.session_state["selected_results"]
    # Avoid duplicates by asset_id
    item_id = item.get("asset_id")
    if any(s.get("asset_id") == item_id for s in sel):
        return  # already selected
    if len(sel) >= 2:
        sel.pop(0)  # remove oldest to make room
    sel.append(item)


# ── Display Results (defined first so it can be called later) ────────────

def sort_results_by_similarity(results: list[dict], ascending: bool = False) -> list[dict]:
    """
    Convert distance to similarity and sort results.
    ascending=False → high to low (default)
    ascending=True  → low to high
    """
    for item in results:
        distance = item.get("distance")
        if distance is not None:
            item["similarity"] = max(0.0, 1.0 - float(distance))
        else:
            item["similarity"] = 0.0
    return sorted(results, key=lambda x: x["similarity"], reverse=not ascending)


def _fmt_time(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def format_manga_info(item: dict) -> str | None:
    """Format manga title and page number into a display string."""
    manga_title = item.get("manga_title")
    page_number = item.get("page_number")
    asset_type = item.get("asset_type", "image")

    if asset_type == "video":
        return format_video_info(item)

    if asset_type == "pdf":
        return format_pdf_info(item)

    if manga_title and page_number is not None:
        return f"📖 {manga_title} — Page {page_number}"
    elif manga_title:
        return f"📖 {manga_title}"
    elif page_number is not None:
        return f"📄 Page {page_number}"
    return None


def format_pdf_info(item: dict) -> str | None:
    """Format PDF document info."""
    manga_title = item.get("manga_title")
    page_number = item.get("page_number")

    if manga_title and page_number is not None:
        return f"📄 {manga_title} — Page {page_number}"
    elif manga_title:
        return f"📄 {manga_title}"
    elif page_number is not None:
        return f"📄 PDF Page {page_number}"
    return None


def format_video_info(item: dict) -> str | None:
    """Format video title and time range, e.g. 🎬 进击的巨人 S1E01 — 0:01:15–0:01:30"""
    manga_title = item.get("manga_title")
    chunk_start = item.get("chunk_start")
    chunk_end = item.get("chunk_end")

    time_range = None
    if chunk_start is not None and chunk_end is not None:
        time_range = f"{_fmt_time(chunk_start)}–{_fmt_time(chunk_end)}"

    if manga_title and time_range:
        return f"🎬 {manga_title} — {time_range}"
    elif manga_title:
        return f"🎬 {manga_title}"
    elif time_range:
        return f"🎬 {time_range}"
    return None


def display_results(results: list[dict], search_type: str):
    """Display search results in a grid with select buttons."""

    if search_type == "Images":
        cols = st.columns(3)
        for i, item in enumerate(results):
            with cols[i % 3]:
                file_url = file_url_for(item)

                st.image(file_url, use_container_width=True)

                manga_info = format_manga_info(item)
                if manga_info:
                    st.caption(manga_info)

                if item.get("dialogue_text"):
                    st.markdown(f"**🇯🇵 {item['dialogue_text']}**")
                if item.get("translation_en"):
                    st.caption(f"🇺🇸 {item['translation_en']}")

                similarity = item.get("similarity")
                if similarity is not None:
                    st.caption(f"🎯 Similarity: {similarity:.2%}")

                # Select button — use index i to guarantee unique keys
                if st.button(f"✅ Select", key=f"select_{search_type}_{i}_{item.get('asset_id', '')}"):
                    add_selected(item)
                    st.rerun()
                if st.button("📖 Study", key=f"study_{search_type}_{i}_{item.get('asset_id', '')}"):
                    st.session_state["study_jump"] = {
                        "manga_id": item.get("manga_id"),
                        "asset_id": item.get("asset_id"),
                    }
                    st.toast("Open the 📖 Study tab to read this page")

    elif search_type == "Audio":
        for i, item in enumerate(results):
            file_url = file_url_for(item)

            col1, col2, col3 = st.columns([1, 2, 0.5])
            with col1:
                st.audio(file_url)
            with col2:
                manga_info = format_manga_info(item)
                if manga_info:
                    st.caption(manga_info)
                if item.get("dialogue_text"):
                    st.markdown(f"**🇯🇵 {item['dialogue_text']}**")
                if item.get("translation_en"):
                    st.caption(f"🇺🇸 {item['translation_en']}")
                similarity = item.get("similarity")
                if similarity is not None:
                    st.caption(f"🎯 Similarity: {similarity:.2%}")
            with col3:
                if st.button(f"✅ Select", key=f"select_{search_type}_{i}_{item.get('asset_id', '')}"):
                    add_selected(item)
                    st.rerun()
                if st.button("📖 Study", key=f"study_{search_type}_{i}_{item.get('asset_id', '')}"):
                    st.session_state["study_jump"] = {
                        "manga_id": item.get("manga_id"),
                        "asset_id": item.get("asset_id"),
                    }
                    st.toast("Open the 📖 Study tab to read this page")

    elif search_type == "Video":
        cols = st.columns(3)
        for i, item in enumerate(results):
            with cols[i % 3]:
                file_url = file_url_for(item)

                st.video(file_url)

                manga_info = format_manga_info(item)
                if manga_info:
                    st.caption(manga_info)

                if item.get("dialogue_text"):
                    st.markdown(f"**🇯🇵 {item['dialogue_text']}**")
                if item.get("translation_en"):
                    st.caption(f"🇺🇸 {item['translation_en']}")

                similarity = item.get("similarity")
                if similarity is not None:
                    st.caption(f"🎯 Similarity: {similarity:.2%}")

                if st.button(f"✅ Select", key=f"select_{search_type}_{i}_{item.get('asset_id', '')}"):
                    add_selected(item)
                    st.rerun()
                if st.button("📖 Study", key=f"study_{search_type}_{i}_{item.get('asset_id', '')}"):
                    st.session_state["study_jump"] = {
                        "manga_id": item.get("manga_id"),
                        "asset_id": item.get("asset_id"),
                    }
                    st.toast("Open the 📖 Study tab to read this page")

    elif search_type == "PDF":
        for i, item in enumerate(results):
            with st.container():
                col1, col2 = st.columns([4, 0.5])
                with col1:
                    manga_info = format_manga_info(item)
                    if manga_info:
                        st.markdown(f"### {manga_info}")

                    dialogue_text = item.get("dialogue_text", "")
                    if dialogue_text:
                        st.markdown(f"> {dialogue_text}")

                    similarity = item.get("similarity")
                    if similarity is not None:
                        st.caption(f"🎯 Similarity: {similarity:.2%}")
                with col2:
                    if st.button(f"✅ Select", key=f"select_{search_type}_{i}_{item.get('asset_id', '')}"):
                        add_selected(item)
                        st.rerun()
                st.divider()

# ── Page Header ──────────────────────────────────────────────────────────

st.title("📚 Manga-Lingo")
st.caption("Search manga to learn Japanese & Korean — powered by Gemini Embedding 2")


# ── Selected Results Panel (up to 2 side by side) ────────────────────────

selected_items = st.session_state.get("selected_results", [])
if selected_items:
    count = len(selected_items)
    st.subheader(f"🎯 Selected Result{'s' if count > 1 else ''} ({count}/2)")

    # Use 4 columns so each item takes ~25% width (50% smaller than before)
    sel_cols = st.columns(4)
    for idx, selected in enumerate(selected_items):
        with sel_cols[idx]:
            file_url = file_url_for(selected)
            asset_type = selected.get("asset_type", "image")

            if asset_type == "image":
                st.image(file_url, use_container_width=True)
            elif asset_type == "audio":
                st.audio(file_url)
            elif asset_type == "video":
                st.video(file_url)
            elif asset_type == "pdf":
                st.markdown("📄 **PDF Document**")

            manga_info = format_manga_info(selected)
            if manga_info:
                st.markdown(f"**{manga_info}**")
            if selected.get("dialogue_text"):
                st.markdown(f"🇯🇵 **{selected['dialogue_text']}**")
            if selected.get("translation_en"):
                st.markdown(f"🇺🇸 {selected['translation_en']}")
            similarity = selected.get("similarity")
            if similarity is not None:
                st.markdown(f"🎯 Similarity: **{similarity:.2%}**")
            st.caption(f"Asset ID: {selected.get('asset_id', 'N/A')}")

            if st.button(f"❌ Remove", key=f"remove_sel_{idx}"):
                st.session_state["selected_results"].pop(idx)
                st.rerun()

    # Clear all button
    if st.button("🗑️ Clear All Selections"):
        st.session_state["selected_results"] = []
        st.rerun()
    st.divider()


# ── Sidebar: Health & Ingestion ──────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    # Health check — first thing in sidebar
    if st.button("🔌 Check API Connection"):
        try:
            r = requests.get(f"{API_BASE}/api/health", timeout=5)
            data = r.json()
            if data.get("db_connected"):
                st.success("✅ API & Database connected")
            else:
                st.warning("⚠️ API up, but DB not connected")
        except Exception as e:
            st.error(f"❌ API not reachable: {e}")

    st.divider()

    # Embedding dimension selector
    st.subheader("🔢 Embedding Dimension")
    embedding_dim = st.radio(
        "Select dimension for search & ingest",
        options=[768, 1536, 3072],
        index=0,
        help=(
            "**768** — Fast, lower memory, good baseline\n\n"
            "**1536** — Balanced quality vs. cost\n\n"
            "**3072** — Highest quality, more compute & storage"
        ),
        key="embedding_dim",
    )

    st.divider()

    # ── Ingest Section ───────────────────────────────────────────────────
    st.header("📥 Ingest Content")

    # Create manga
    with st.expander("➕ Create Manga"):
        manga_title = st.text_input("Manga Title")
        manga_lang = st.selectbox("Language", ["ja", "ko"])
        if st.button("Create Manga"):
            if manga_title:
                try:
                    r = requests.post(
                        f"{API_BASE}/api/manga",
                        json={"title": manga_title, "language": manga_lang},
                    )
                    st.success(f"Created manga: {r.json()}")
                except Exception as e:
                    st.error(str(e))

    # Upload asset
    with st.expander("📤 Upload Asset"):
        upload_file = st.file_uploader(
            "Choose a file",
            type=["png", "jpg", "jpeg", "webp", "mp3", "ogg", "wav", "mp4", "webm"],
        )
        upload_type = st.selectbox("Asset Type", ["image", "audio", "video"])
        upload_manga_id = st.number_input("Manga ID (optional)", min_value=0, value=0, step=1)
        upload_dialogue = st.text_input("Dialogue Text (JP/KR)", value="")
        upload_translation = st.text_input("English Translation", value="")

        if st.button("Upload & Ingest"):
            if upload_file:
                try:
                    files = {"file": (upload_file.name, upload_file.getvalue())}
                    form_data = {
                        "asset_type": upload_type,
                        "embedding_dim": str(embedding_dim),
                    }
                    if upload_manga_id > 0:
                        form_data["manga_id"] = str(upload_manga_id)
                    if upload_dialogue:
                        form_data["dialogue_text"] = upload_dialogue
                    if upload_translation:
                        form_data["translation_en"] = upload_translation

                    with st.spinner(f"Generating {embedding_dim}d embedding & saving..."):
                        r = requests.post(f"{API_BASE}/api/ingest", files=files, data=form_data)
                        if r.status_code == 200:
                            st.success(f"✅ Ingested: {r.json()}")
                        else:
                            st.error(f"Error: {r.text}")
                except Exception as e:
                    st.error(str(e))
            else:
                st.warning("Please select a file first")

    # Upload PDF
    with st.expander("📄 Upload PDF"):
        pdf_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            key="pdf_upload",
        )
        pdf_manga_id = st.number_input("Manga ID (optional)", min_value=0, value=0, step=1, key="pdf_manga_id")
        pdf_manga_title = st.text_input("Title (optional)", value="", key="pdf_manga_title")

        if st.button("Upload & Ingest PDF"):
            if pdf_file:
                try:
                    files = {"file": (pdf_file.name, pdf_file.getvalue())}
                    form_data = {
                        "embedding_dim": str(embedding_dim),
                    }
                    if pdf_manga_id > 0:
                        form_data["manga_id"] = str(pdf_manga_id)
                    if pdf_manga_title:
                        form_data["manga_title"] = pdf_manga_title

                    with st.spinner(f"Extracting text & generating {embedding_dim}d embeddings per page..."):
                        r = requests.post(f"{API_BASE}/api/ingest/pdf", files=files, data=form_data)
                        if r.status_code == 200:
                            result = r.json()
                            st.success(f"✅ Ingested {result.get('pages_ingested', 0)} pages")
                        else:
                            st.error(f"Error: {r.text}")
                except Exception as e:
                    st.error(str(e))
            else:
                st.warning("Please select a PDF file first")


# ── Main: Search ─────────────────────────────────────────────────────────

tab_text, tab_image, tab_audio, tab_compare, tab_study, tab_review = st.tabs(
    ["🔍 Text Search", "🖼️ Image Search", "🎧 Audio Search", "⚖️ Compare Dimensions", "📖 Study", "🧠 Review"]
)

# ── Text Search Tab ──────────────────────────────────────────────────────
with tab_text:
    # Use a form so pressing Enter submits the search
    with st.form(key="text_search_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Search query",
                placeholder="e.g., angry samurai, ありがとう, 전투 장면...",
            )
        with col2:
            search_type = st.selectbox("Search for", ["Images", "Audio", "Video", "PDF"])
            limit = st.slider("Max results", 1, 50, 10)

        submitted = st.form_submit_button("🔍 Search")

    if submitted and query:
        endpoint_map = {
            "Images": "text-to-image",
            "Audio": "text-to-audio",
            "Video": "text-to-video",
            "PDF": "text-to-pdf",
        }
        endpoint = endpoint_map[search_type]

        with st.spinner(f"Searching {search_type.lower()}..."):
            try:
                r = requests.post(
                    f"{API_BASE}/api/search/{endpoint}",
                    json={"query": query, "limit": limit, "embedding_dim": embedding_dim},
                )
                results = r.json().get("results", [])

                if not results:
                    st.info("No results found. Try ingesting some content first!")
                else:
                    # Store results in session state for sorting
                    st.session_state["text_search_results"] = results
                    st.session_state["text_search_type"] = search_type

            except Exception as e:
                st.error(f"Search failed: {e}")

    # Display results with sort toggle (outside the form so it works without re-searching)
    if "text_search_results" in st.session_state and st.session_state["text_search_results"]:
        results = st.session_state["text_search_results"]
        s_type = st.session_state.get("text_search_type", "Images")

        sort_col1, sort_col2 = st.columns([3, 1])
        with sort_col1:
            st.success(f"Found {len(results)} results")
        with sort_col2:
            sort_order = st.selectbox(
                "Sort by similarity",
                ["High → Low", "Low → High"],
                key="text_sort_order",
            )

        ascending = sort_order == "Low → High"
        sorted_results = sort_results_by_similarity(results, ascending=ascending)
        display_results(sorted_results, s_type)

# ── Image Search Tab ─────────────────────────────────────────────────────
with tab_image:
    uploaded_image = st.file_uploader(
        "Upload an image to search for similar content",
        type=["png", "jpg", "jpeg", "webp"],
        key="image_search_upload",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        image_search_type = st.selectbox(
            "Search for", ["Similar Images", "Related Videos"], key="img_search_type"
        )
    with col2:
        image_limit = st.slider("Max results", 1, 50, 10, key="img_limit")

    if uploaded_image:
        st.image(uploaded_image, caption="Query Image", width=300)

        if st.button("🔍 Search by Image", key="image_search"):
            endpoint = (
                "image-to-image"
                if image_search_type == "Similar Images"
                else "image-to-video"
            )

            with st.spinner("Searching..."):
                try:
                    files = {"file": (uploaded_image.name, uploaded_image.getvalue())}
                    data = {"limit": str(image_limit), "embedding_dim": str(embedding_dim)}
                    r = requests.post(
                        f"{API_BASE}/api/search/{endpoint}", files=files, data=data
                    )
                    results = r.json().get("results", [])

                    if not results:
                        st.info("No results found.")
                    else:
                        result_type = (
                            "Images"
                            if image_search_type == "Similar Images"
                            else "Video"
                        )
                        st.session_state["image_search_results"] = results
                        st.session_state["image_search_result_type"] = result_type

                except Exception as e:
                    st.error(f"Search failed: {e}")

    # Display image search results with sort toggle
    if "image_search_results" in st.session_state and st.session_state["image_search_results"]:
        results = st.session_state["image_search_results"]
        r_type = st.session_state.get("image_search_result_type", "Images")

        sort_col1, sort_col2 = st.columns([3, 1])
        with sort_col1:
            st.success(f"Found {len(results)} results")
        with sort_col2:
            img_sort_order = st.selectbox(
                "Sort by similarity",
                ["High → Low", "Low → High"],
                key="img_sort_order",
            )

        ascending = img_sort_order == "Low → High"
        sorted_results = sort_results_by_similarity(results, ascending=ascending)
        display_results(sorted_results, r_type)

# ── Audio Search Tab ─────────────────────────────────────────────────────
with tab_audio:
    st.caption(
        "Upload a few seconds of audio — a line of dialogue, a scream, the opening song — "
        "to find the anime scene. It is matched against each chunk's audio track, so a "
        "phone recording of the TV works too."
    )
    uploaded_audio = st.file_uploader(
        "Upload an audio clip",
        type=["mp3", "wav", "m4a", "aac", "ogg", "flac"],
        key="audio_search_upload",
    )
    audio_limit = st.slider("Max results", 1, 50, 6, key="audio_limit")

    if uploaded_audio:
        st.audio(uploaded_audio)
        if st.button("🔍 Search by Audio", key="audio_search"):
            with st.spinner("Searching..."):
                try:
                    files = {"file": (uploaded_audio.name, uploaded_audio.getvalue(), uploaded_audio.type)}
                    data = {"limit": str(audio_limit), "embedding_dim": str(embedding_dim)}
                    r = requests.post(f"{API_BASE}/api/search/audio-to-video", files=files, data=data)
                    results = r.json().get("results", [])
                    st.session_state["audio_search_results"] = results
                    if not results:
                        st.info("No results found.")
                except Exception as e:
                    st.error(f"Search failed: {e}")

    if st.session_state.get("audio_search_results"):
        results = st.session_state["audio_search_results"]
        st.success(f"Found {len(results)} results")
        display_results(sort_results_by_similarity(results), "Video")

# ── Compare Dimensions Tab ───────────────────────────────────────────────
with tab_compare:
    st.markdown(
        "Run the **same query** against multiple embedding dimensions side-by-side "
        "to see how 768 vs 1536 vs 3072 affect search quality."
    )

    with st.form(key="compare_form"):
        cmp_col1, cmp_col2 = st.columns([3, 1])
        with cmp_col1:
            cmp_query = st.text_input(
                "Search query",
                placeholder="e.g., titan attacking the wall, 巨人が壁を攻撃する...",
                key="cmp_query",
            )
        with cmp_col2:
            cmp_search_type = st.selectbox(
                "Search for", ["Images", "Video", "PDF"], key="cmp_search_type_input"
            )
            cmp_limit = st.slider("Results per dimension", 1, 20, 5, key="cmp_limit")

        cmp_dims = st.multiselect(
            "Dimensions to compare",
            options=[768, 1536, 3072],
            default=[768, 1536, 3072],
            key="cmp_dims",
        )

        cmp_submitted = st.form_submit_button("⚖️ Compare")

    if cmp_submitted and cmp_query and cmp_dims:
        endpoint_map = {
            "Images": "text-to-image",
            "Video": "text-to-video",
            "PDF": "text-to-pdf",
        }
        endpoint = endpoint_map[cmp_search_type]

        all_results = {}
        with st.spinner(f"Searching across {len(cmp_dims)} dimensions..."):
            for dim in sorted(cmp_dims):
                try:
                    r = requests.post(
                        f"{API_BASE}/api/search/{endpoint}",
                        json={"query": cmp_query, "limit": cmp_limit, "embedding_dim": dim},
                    )
                    results = r.json().get("results", [])
                    all_results[dim] = sort_results_by_similarity(results)
                except Exception as e:
                    st.error(f"Search failed for dim={dim}: {e}")
                    all_results[dim] = []

        st.session_state["cmp_results"] = all_results
        st.session_state["cmp_result_type"] = cmp_search_type

    # Display comparison results
    if "cmp_results" in st.session_state and st.session_state["cmp_results"]:
        all_results = st.session_state["cmp_results"]
        cmp_s_type = st.session_state.get("cmp_result_type", "Images")
        dims = sorted(all_results.keys())

        # Create side-by-side columns, one per dimension
        dim_cols = st.columns(len(dims))

        for col_idx, dim in enumerate(dims):
            with dim_cols[col_idx]:
                results = all_results[dim]
                st.subheader(f"🔢 {dim}d")
                if not results:
                    st.info(f"No results for {dim}d. Have you ingested with --embedding-dim {dim}?")
                    continue

                st.caption(f"{len(results)} results")

                for i, item in enumerate(results):
                    file_url = file_url_for(item)
                    similarity = item.get("similarity")

                    if cmp_s_type == "Images":
                        st.image(file_url, use_container_width=True)
                    elif cmp_s_type == "Video":
                        st.video(file_url)
                    elif cmp_s_type == "PDF":
                        dialogue_text = item.get("dialogue_text", "")
                        if dialogue_text:
                            st.markdown(f"> {dialogue_text[:200]}")

                    manga_info = format_manga_info(item)
                    if manga_info:
                        st.caption(manga_info)

                    if similarity is not None:
                        st.caption(f"🎯 {similarity:.2%}")

                    st.divider()


# ── Study Tab ────────────────────────────────────────────────────────────
with tab_study:
    render_study_tab(file_url_for)

# ── Review Tab ───────────────────────────────────────────────────────────
with tab_review:
    render_review_tab(file_url_for)
