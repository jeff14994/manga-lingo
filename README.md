# 📚 Manga-Lingo

A multimodal search app for learning Japanese & Korean through manga — powered by **Gemini Embedding 2 Preview** (`gemini-embedding-2-preview`) and **PostgreSQL + pgvector** (AlloyDB optional).

Read a manga page → find the matching anime scene → hear it in Japanese, with furigana, word meanings and spaced-repetition review.

## 🎤 PyCon TW 2026

**Learn Japanese with Manga! Building a Multimodal Search Engine with Gemini Embedding 2** — 10:50–11:20 · 中階 · 中文（英文簡報）· @x0mg

- Talk: [PyCon TW 2026 · Day 2 (Sun, Oct 18) · 10:50–11:20 · R0](https://tw.pycon.org/2026/en-us/conference/talk/378)
- Slides: [`presentation/pycon2026.pdf`](presentation/pycon2026.pdf) (PyCon TW 2026)
- Evaluation: [`eval/queries.yaml`](eval/queries.yaml) (44 hand-labeled queries) and [`eval/results.md`](eval/results.md)

**Does 3072 beat 768?** Not on this data:

| Dim | Recall@10 (pooled) | MRR | Search p50 | Bytes / vector |
|---:|---:|---:|---:|---:|
| 768 | 0.96 | 0.88 | 5.3 ms | 3 KB |
| 1536 | 0.93 | 0.88 | 6.8 ms | 6 KB |
| 3072 | 0.94 | 0.88 | 12.1 ms | 12 KB |

Quality is within noise; 3072 costs 4× the storage and ~2.3× the search latency. Hard queries (abstract emotions, small details) lower MRR for every dimension alike. For audio search, match against each chunk's **audio track** (`scripts/index_audio_tracks.py`), not the video embedding: a video embedding is dominated by its visuals, and audio MRR goes from 0.66 to 0.83. Measure on your own data — see [Reproduce the evaluation](#reproduce-the-dimension-evaluation).

> **Data & copyright:** this repo contains code and labels only — no manga scans or video. Use media you own or that is licensed for reuse (e.g. 『ブラックジャックによろしく』, free for secondary use).

## Quick Start

### 1. Install dependencies

```bash
cd manga-lingo
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Database: local PostgreSQL (recommended) or AlloyDB

**Local PostgreSQL + pgvector** — e.g. [Postgres.app](https://postgresapp.com/) on macOS (pgvector included). In `.env`:

```env
ALLOYDB_HOST=127.0.0.1
ALLOYDB_PORT=5432
ALLOYDB_USER=postgres
ALLOYDB_PASSWORD=
ALLOYDB_DATABASE=postgres
ALLOYDB_SSLMODE=disable
```

**Or AlloyDB** — point the same `ALLOYDB_*` variables at your instance and set `ALLOYDB_SSLMODE=require`.

### 3. Set up the database (one-time)

```bash
psql -h 127.0.0.1 -p 5432 -U postgres -d postgres -f scripts/setup_db.sql
```

### 4. Start the backend

```bash
# Binds to localhost only; don't expose the API on 0.0.0.0
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Verify it works:
- **Quick ping (no DB):** http://localhost:8000/api/ping → `{"status":"ok"}`
- **Health check (needs DB):** http://localhost:8000/api/health → `{"status":"ok","db_connected":true}`
- **API docs:** http://localhost:8000/docs

### 5. Start the frontend

```bash
# New terminal (activate venv first)
source .venv/bin/activate
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

### 6. Seed sample data (optional)

```bash
# With the backend running, in another terminal:
source .venv/bin/activate
python scripts/seed_data.py
```

This generates **12 sample manga panels** across 3 manga titles (Demon Slayer, Solo Leveling, One Piece) with Japanese/Korean dialogue, English translations, and Gemini embeddings.

After seeding, try searching:
- `water breathing sword`
- `海賊王` (Pirate King)
- `dark power level up`
- `仲間を守る` (protect friends)
- `전투` (battle)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/ping` | Quick health check (no DB) |
| `GET` | `/api/health` | Full health check with DB connection test |
| `POST` | `/api/search/text-to-image` | Search images by text query |
| `POST` | `/api/search/text-to-audio` | Search audio by text query |
| `POST` | `/api/search/text-to-video` | Search video by text query |
| `POST` | `/api/search/image-to-image` | Search similar images by image upload |
| `POST` | `/api/search/image-to-video` | Search related videos by image upload |
| `POST` | `/api/search/audio-to-video` | Search video scenes by an audio clip |
| `GET` | `/api/assets/{id}/scenes` | Anime scenes matching a manga page (stored embedding) |
| `GET` | `/api/assets/{id}/lines` | Dialogue lines with furigana, words, JLPT levels |
| `GET` | `/api/words/{lemma}/occurrences` | Where a word appears in manga / anime |
| `GET` / `POST` | `/api/review/...` | Spaced-repetition review (due, add, grade) |
| `POST` | `/api/manga` | Create a manga entry |
| `GET` | `/api/manga` | List all manga entries |
| `POST` | `/api/ingest` | Upload & ingest an asset (image/audio/video) |

Full interactive docs at http://localhost:8000/docs

## Features

- **Text → Image/Audio/Video search** — type a query, find matching manga content
- **Image → Image/Video search** — upload an image, find similar panels or related clips
- **Ingestion** — upload manga images/audio/video with metadata, auto-generates embeddings
- **Multimodal embeddings** — all modalities share the same embedding space via `gemini-embedding-2-preview`
- **📖 Study tab** — read a page bubble by bubble with furigana, word meanings, JLPT levels and grammar notes
- **Manga page → anime scene** — the page's stored embedding finds the matching anime clip, shown with its Japanese transcript (no shared words needed)
- **🧠 Review tab** — spaced-repetition word cards that show the panel and the anime moment where the word is spoken

### Japanese study setup

```bash
# Extract speech bubbles (Japanese manga) and transcribe anime clips with Gemini
python scripts/extract_dialogue.py

# Add 1536/3072 embeddings to already-ingested assets (for the Compare tab)
python scripts/reembed.py --dim 1536 3072
```

Credits: dictionary glosses from [JMdict](https://www.edrdg.org/jmdict/j_jmdict.html)
(© EDRDG, CC BY-SA 4.0) via jamdict; JLPT levels from Jonathan Waller's
[JLPT lists](https://www.tanos.co.uk/jlpt/) (CC BY) via
[elzup/jlpt-word-list](https://github.com/elzup/jlpt-word-list).

## Reproduce the dimension evaluation

```bash
# 1. Make sure every asset has 768 / 1536 / 3072 embeddings
python scripts/reembed.py --dim 1536 3072

# 2. Write your queries in eval/queries.yaml, then pool the top-10 of every
#    dimension into a labeling page and tick the correct results
python scripts/eval_retrieval.py draft
open eval/label.html          # Generate → Copy → paste over eval/queries.yaml

# 3. Score: Recall@10, MRR, latency and bytes per vector by dimension
python scripts/eval_retrieval.py run   # → eval/results.md
```

Query types: `text_to_image`, `text_to_video`, `image_to_video`, `audio_to_video`.
Labels are pooled (judged from the union of all dimensions' top-10), so Recall@10
is optimistic — compare dimensions with MRR.

## Project Structure

```
manga-lingo/
├── backend/
│   ├── main.py              # FastAPI app & endpoints
│   ├── config.py            # Environment config (.env loader)
│   ├── db.py                # PostgreSQL connection pool (psycopg2)
│   ├── data/jlpt_vocab.csv  # JLPT levels (Tanos lists, CC BY)
│   └── services/
│       ├── vertex.py        # Shared Vertex AI auth / retry
│       ├── embedding.py     # Gemini Embedding 2
│       ├── gemini.py        # Gemini generateContent (JSON output)
│       ├── search.py        # Vector search (halfvec for 3072)
│       ├── ingest.py        # File save + embed + DB insert
│       ├── dialogue.py      # Speech bubbles & transcripts
│       ├── japanese.py      # Furigana, readings, JMdict, JLPT
│       └── study.py         # Study / review (SM-2)
├── frontend/
│   ├── app.py               # Streamlit UI
│   └── study_ui.py          # Study & Review tabs
├── scripts/                 # setup_db.sql, ingest, re-embed, audio tracks, eval
├── eval/                    # 44 labeled queries + results
├── presentation/
│   └── pycon2026.pdf        # PyCon TW 2026 slides
├── data/                    # Local media (gitignored)
├── .env                     # Secrets (gitignored)
└── requirements.txt
```

## Architecture

```
Streamlit (8501) → FastAPI (8000) → Gemini Embedding 2 (Vertex AI)
                                  → PostgreSQL + pgvector (local, or AlloyDB)
```

## Environment Variables (`.env`)

```env
# Google Cloud
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS_JSON={...service account JSON...}

# Gemini for dialogue extraction / transcription (defaults shown)
GEMINI_MODEL=gemini-3-flash-preview
GEMINI_LOCATION=global

# Database (local PostgreSQL shown; for AlloyDB use its host and ALLOYDB_SSLMODE=require)
ALLOYDB_HOST=127.0.0.1
ALLOYDB_PORT=5432
ALLOYDB_USER=postgres
ALLOYDB_PASSWORD=
ALLOYDB_DATABASE=postgres
ALLOYDB_SSLMODE=disable
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ERR_CONNECTION_REFUSED` in browser | Backend isn't running yet — start uvicorn and wait ~20s |
| `/api/health` returns `db_error` | PostgreSQL isn't running (start Postgres.app) or `ALLOYDB_*` in `.env` is wrong |
| `python-dotenv could not parse` | `.env` must use `KEY=VALUE` format (not `"KEY": "VALUE"`) |
| `uvicorn: command not found` | Activate venv first: `source .venv/bin/activate` |

## License

Code: [MIT](LICENSE). Dictionary data: JMdict (EDRDG, CC BY-SA 4.0); JLPT levels: Jonathan Waller's lists (tanos.co.uk, CC BY).
