#!/usr/bin/env python3
"""
Retrieval evaluation: Recall@10 and MRR for 768 / 1536 / 3072 dimensions.

Two steps:

    # 1. Pool the top-10 results of every query at every dimension into a
    #    labeling page. Tick the correct answers, then copy the generated
    #    YAML back into eval/queries.yaml.
    python scripts/eval_retrieval.py draft
    open eval/label.html

    # 2. Score every labeled query at each dimension.
    python scripts/eval_retrieval.py run

Pooling (judging the union of all systems' top-k) is the standard way to
build relevance labels without judging every item in the collection; an
item no dimension ever returned can't be judged, which is fine for
comparing the dimensions against each other.

Metrics:
    Recall@10  = relevant items in the top 10 / all relevant items
    MRR        = mean of 1 / rank of the first relevant result (0 if none)
Latency is measured separately for the query embedding (Gemini API call)
and the vector search (PostgreSQL).
"""

import argparse
import html
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db import get_db  # noqa: E402
from backend.services.embedding import (  # noqa: E402
    ALLOWED_EMBEDDING_DIMS,
    embed_audio_from_path,
    embed_image_from_path,
    embed_text,
)
from backend.services.search import _vector_search, audio_embedding_to_video  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval"
QUERIES_FILE = EVAL_DIR / "queries.yaml"
LABEL_HTML = EVAL_DIR / "label.html"
THUMB_DIR = EVAL_DIR / "thumbs"
RESULTS_MD = EVAL_DIR / "results.md"
TOP_K = 10

TARGET_MODALITY = {
    "text_to_image": "image",
    "text_to_video": "video",
    "image_to_video": "video",
    "audio_to_video": "video",
}


# ── Helpers ──────────────────────────────────────────────────────────────


def _mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _parse_mmss(text: str) -> float:
    m, s = text.strip().split(":")
    return int(m) * 60 + float(s)


def result_key(item: dict) -> str:
    """Stable label for a search result, independent of asset_id."""
    if item["asset_type"] == "video":
        return f"{item['manga_title']}|{_mmss(item['chunk_start'])}-{_mmss(item['chunk_end'])}"
    return f"{item['manga_title']}|p{item['page_number']}"


def is_relevant(item: dict, relevant: list[str]) -> bool:
    """Image: exact page key. Video: chunk overlaps any labeled time range."""
    if item["asset_type"] != "video":
        return result_key(item) in relevant
    for label in relevant:
        title, _, span = label.rpartition("|")
        if title != item["manga_title"] or "-" not in span:
            continue
        start, end = (_parse_mmss(t) for t in span.split("-"))
        if item["chunk_start"] < end and item["chunk_end"] > start:
            return True
    return False


def embed_query(q: dict, dim: int) -> list[float]:
    if q["type"] == "image_to_video":
        return embed_image_from_path(str(PROJECT_ROOT / q["query"]), dim=dim)
    if q["type"] == "audio_to_video":
        return embed_audio_from_path(str(PROJECT_ROOT / q["query"]), dim=dim)
    return embed_text(q["query"], dim=dim)


def run_query(q: dict, dim: int, limit: int = TOP_K) -> tuple[list[dict], float, float]:
    """Return (results, embed_ms, search_ms)."""
    t0 = time.perf_counter()
    embedding = embed_query(q, dim)
    t1 = time.perf_counter()
    if q["type"] == "audio_to_video":
        # Same path as the API: match audio tracks, return their video chunks
        results = audio_embedding_to_video(embedding, limit=limit, embedding_dim=dim)
    else:
        results = _vector_search(embedding, TARGET_MODALITY[q["type"]], limit=limit, embedding_dim=dim)
    t2 = time.perf_counter()
    return results, (t1 - t0) * 1000, (t2 - t1) * 1000


def load_queries() -> list[dict]:
    data = yaml.safe_load(QUERIES_FILE.read_text(encoding="utf-8"))
    return data["queries"]


def _video_thumb(item: dict) -> str:
    """Extract one frame from the middle of a video chunk (cached)."""
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    out = THUMB_DIR / f"{item['asset_id']}.jpg"
    if not out.exists():
        mid = max(0.0, (item["chunk_end"] - item["chunk_start"]) / 2)
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{mid:.1f}",
             "-i", str(PROJECT_ROOT / item["file_path"]),
             "-frames:v", "1", "-vf", "scale=320:-1", str(out)],
            check=False,
        )
    return f"thumbs/{out.name}"


# ── draft: build the labeling page ───────────────────────────────────────


def cmd_draft(args):
    queries = load_queries()
    pools = []
    for q in queries:
        pooled: dict[str, dict] = {}
        for dim in args.dims:
            results, _, _ = run_query(q, dim)
            for rank, item in enumerate(results, 1):
                key = result_key(item)
                entry = pooled.setdefault(key, {"item": item, "ranks": {}})
                entry["ranks"][dim] = rank
        print(f"  {q['id']}: {len(pooled)} pooled candidates")
        pools.append((q, pooled))

    LABEL_HTML.write_text(_label_page(pools), encoding="utf-8")
    print(f"\n✅ Wrote {LABEL_HTML.relative_to(PROJECT_ROOT)} — open it, tick the correct results, "
          f"then paste the generated YAML into {QUERIES_FILE.relative_to(PROJECT_ROOT)}")


def _label_page(pools) -> str:
    blocks = []
    for q, pooled in pools:
        cards = []
        # Best rank across dimensions first
        for key, entry in sorted(pooled.items(), key=lambda kv: min(kv[1]["ranks"].values())):
            item = entry["item"]
            if item["asset_type"] == "video":
                src = _video_thumb(item)
            else:
                src = "../" + item["file_path"]
            ranks = " ".join(f"{d}:#{r}" for d, r in sorted(entry["ranks"].items()))
            checked = "checked" if is_relevant(item, q.get("relevant") or []) else ""
            # One mid-clip frame can mislead; let the labeler play the whole chunk
            play = (
                f' <a href="../{html.escape(item["file_path"])}" target="_blank">▶ play</a>'
                if item["asset_type"] == "video" else ""
            )
            cards.append(
                f'<label class="card"><img src="{html.escape(src)}" loading="lazy">'
                f'<input type="checkbox" data-q="{q["id"]}" value="{html.escape(key)}" {checked}>'
                f'<span>{html.escape(key)}{play}</span><small>{ranks}</small></label>'
            )
        if q["type"] == "image_to_video":
            query_view = f'<img class="qimg" src="../{html.escape(q["query"])}">'
        elif q["type"] == "audio_to_video":
            query_view = f'<audio controls src="../{html.escape(q["query"])}"></audio>'
        else:
            query_view = f"<b>{html.escape(q['query'])}</b>"
        blocks.append(
            f'<section><h2>{q["id"]} <small>{q["type"]}</small></h2>{query_view}'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )

    queries_json = json.dumps(
        [{k: q[k] for k in ("id", "type", "query")} for q, _ in pools], ensure_ascii=False
    )
    return f"""<!doctype html><meta charset="utf-8"><title>manga-lingo retrieval labels</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #111; color: #eee; }}
section {{ border-top: 1px solid #444; padding: 1rem 0; }}
.grid {{ display: flex; flex-wrap: wrap; gap: .6rem; margin-top: .6rem; }}
.card {{ width: 170px; display: flex; flex-direction: column; gap: .2rem; background: #222;
         padding: .4rem; border-radius: .4rem; cursor: pointer; font-size: .8rem; }}
.card:has(input:checked) {{ outline: 3px solid #4caf50; }}
.card img {{ width: 100%; height: 220px; object-fit: contain; background: #000; }}
.qimg {{ height: 260px; }}
textarea {{ width: 100%; height: 22rem; background: #000; color: #8f8; font-family: monospace; }}
#bar {{ position: sticky; top: 0; background: #111; padding: .5rem 0; z-index: 1; }}
</style>
<div id="bar"><button onclick="build()">Generate queries.yaml</button>
<button onclick="navigator.clipboard.writeText(document.getElementById('out').value)">Copy</button>
<small>Video: ticking clips adjacent in time merges them into one range.</small></div>
<textarea id="out" placeholder="Click 'Generate queries.yaml', then paste over eval/queries.yaml"></textarea>
{''.join(blocks)}
<script>
const QUERIES = {queries_json};
const toSec = t => {{ const [m, s] = t.split(':'); return (+m) * 60 + (+s); }};
const toMmss = s => `${{Math.floor(s / 60)}}:${{String(Math.round(s % 60)).padStart(2, '0')}}`;
const q = s => JSON.stringify(s);
function relevantFor(id) {{
  const keys = [...document.querySelectorAll(`input[data-q="${{id}}"]:checked`)].map(i => i.value);
  const pages = keys.filter(k => !/\\|\\d+:\\d+-/.test(k));
  // merge adjacent/overlapping video ranges per title
  const byTitle = {{}};
  keys.filter(k => /\\|\\d+:\\d+-/.test(k)).forEach(k => {{
    const i = k.lastIndexOf('|'); const [a, b] = k.slice(i + 1).split('-').map(toSec);
    (byTitle[k.slice(0, i)] ??= []).push([a, b]);
  }});
  const ranges = [];
  for (const [title, spans] of Object.entries(byTitle)) {{
    spans.sort((x, y) => x[0] - y[0]);
    let [s, e] = spans[0];
    for (const [a, b] of spans.slice(1)) {{ if (a <= e + 1) e = Math.max(e, b); else {{ ranges.push(`${{title}}|${{toMmss(s)}}-${{toMmss(e)}}`); [s, e] = [a, b]; }} }}
    ranges.push(`${{title}}|${{toMmss(s)}}-${{toMmss(e)}}`);
  }}
  return [...pages, ...ranges];
}}
function build() {{
  let y = '# Labeled with eval/label.html (pooled top-{TOP_K} across dimensions)\\nqueries:\\n';
  for (const x of QUERIES) {{
    const rel = relevantFor(x.id);
    y += `  - id: ${{x.id}}\\n    type: ${{x.type}}\\n    query: ${{q(x.query)}}\\n`;
    y += rel.length ? `    relevant:\\n${{rel.map(r => `      - ${{q(r)}}`).join('\\n')}}\\n` : '    relevant: []\\n';
  }}
  document.getElementById('out').value = y;
}}
</script>"""


# ── run: score labeled queries ───────────────────────────────────────────


def cmd_run(args):
    queries = [q for q in load_queries() if q.get("relevant")]
    skipped = len(load_queries()) - len(queries)
    if not queries:
        print("No labeled queries yet. Run `draft`, label eval/label.html, update eval/queries.yaml.")
        return

    chunks = _video_chunks()
    # Warm up credentials + DB pool so the first dimension isn't charged for setup
    run_query({"type": "text_to_image", "query": "warm-up"}, args.dims[0], limit=1)
    rows = []  # (dim, type, recall, rr, embed_ms, search_ms)
    per_query = []
    for q in queries:
        for dim in args.dims:
            results, embed_ms, search_ms = run_query(q, dim)
            hits = [is_relevant(r, q["relevant"]) for r in results[:TOP_K]]
            first = next((i for i, h in enumerate(hits, 1) if h), None)
            rr = 1.0 / first if first else 0.0
            recall = _recall(q, results, chunks)
            rows.append((dim, q["type"], recall, rr, embed_ms, search_ms))
            per_query.append((q["id"], dim, recall, rr, first))
        print(f"  {q['id']}: " + "  ".join(
            f"{d}d R@10={r:.2f} RR={x:.2f}" for i, d, r, x, _ in per_query if i == q["id"]))

    md = _results_markdown(rows, per_query, args.dims, len(queries), skipped)
    RESULTS_MD.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"✅ Saved {RESULTS_MD.relative_to(PROJECT_ROOT)}")


def _video_chunks() -> list[dict]:
    """All video chunks (title + time range), for counting relevant chunks."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT a.asset_type, m.title AS manga_title, a.chunk_start, a.chunk_end
            FROM assets a JOIN manga m ON m.id = a.manga_id
            WHERE a.asset_type = 'video' AND a.chunk_start IS NOT NULL
            """
        )
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
        cursor.close()
    return rows


def _recall(q: dict, results: list[dict], chunks: list[dict]) -> float:
    """
    Recall@10. Image: relevant pages found / labeled pages. Video: relevant
    chunks found / all chunks overlapping the labeled ranges.
    """
    found = sum(is_relevant(r, q["relevant"]) for r in results[:TOP_K])
    if q["type"] == "text_to_image":
        total = len(q["relevant"])
    else:
        total = sum(is_relevant(c, q["relevant"]) for c in chunks)
    return found / total if total else 0.0


def _results_markdown(rows, per_query, dims, n_queries, skipped) -> str:
    def agg(filter_fn):
        sel = [r for r in rows if filter_fn(r)]
        return (statistics.mean(r[2] for r in sel), statistics.mean(r[3] for r in sel),
                statistics.median(r[4] for r in sel), statistics.median(r[5] for r in sel))

    lines = [
        f"## Retrieval quality by embedding dimension ({n_queries} labeled queries"
        + (f", {skipped} unlabeled skipped" if skipped else "") + ")",
        "",
        "| Dim | Recall@10 | MRR | Embed p50 (ms) | Search p50 (ms) | Bytes / vector |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for dim in dims:
        r, m, e, s = agg(lambda row: row[0] == dim)
        lines.append(f"| {dim} | {r:.3f} | {m:.3f} | {e:.0f} | {s:.1f} | {dim * 4:,} |")

    types = sorted({row[1] for row in rows})
    lines += ["", "### By query type", "", "| Type | " + " | ".join(f"{d} R@10 / MRR" for d in dims) + " |",
              "|---|" + "---:|" * len(dims)]
    for t in types:
        cells = []
        for dim in dims:
            r, m, _, _ = agg(lambda row: row[0] == dim and row[1] == t)
            cells.append(f"{r:.2f} / {m:.2f}")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    lines += ["", "<details><summary>Per query (rank of first relevant result)</summary>", "",
              "| Query | " + " | ".join(f"{d}" for d in dims) + " |", "|---|" + "---:|" * len(dims)]
    for qid in dict.fromkeys(p[0] for p in per_query):
        cells = []
        for dim in dims:
            first = next(p[4] for p in per_query if p[0] == qid and p[1] == dim)
            cells.append(f"#{first}" if first else "—")
        lines.append(f"| {qid} | " + " | ".join(cells) + " |")
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Recall@10 / MRR evaluation across embedding dimensions")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("draft", "run"):
        p = sub.add_parser(name)
        p.add_argument("--dims", type=int, nargs="+", default=list(ALLOWED_EMBEDDING_DIMS),
                       choices=ALLOWED_EMBEDDING_DIMS)
    args = parser.parse_args()
    {"draft": cmd_draft, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
