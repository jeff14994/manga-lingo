-- Manga-Lingo: Database Schema Setup
-- Run: psql -h 127.0.0.1 -p 5432 -U postgres -d postgres -f scripts/setup_db.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- Manga metadata
CREATE TABLE IF NOT EXISTS manga (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    language    VARCHAR(5),  -- 'ja', 'ko'
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- All media assets (images, audio, video)
CREATE TABLE IF NOT EXISTS assets (
    id              SERIAL PRIMARY KEY,
    manga_id        INT REFERENCES manga(id),
    asset_type      VARCHAR(10) NOT NULL,  -- 'image', 'audio', 'video'
    file_path       TEXT NOT NULL,          -- local path: ./data/images/001.png
    page_number     INT,                   -- page number within the manga volume
    dialogue_text   TEXT,                   -- original JP/KR text
    translation_en  TEXT,                   -- English translation
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Migration: add page_number if table already exists without it
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'assets' AND column_name = 'page_number'
    ) THEN
        ALTER TABLE assets ADD COLUMN page_number INT;
    END IF;
END $$;

-- Migration: add chunk_start / chunk_end for video time range display
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'assets' AND column_name = 'chunk_start'
    ) THEN
        ALTER TABLE assets ADD COLUMN chunk_start REAL;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'assets' AND column_name = 'chunk_end'
    ) THEN
        ALTER TABLE assets ADD COLUMN chunk_end REAL;
    END IF;
END $$;

-- ── Embeddings (multi-dimension support) ────────────────────────────────
-- Stores embeddings at different dimensions (768, 1536, 3072) for
-- A/B testing embedding quality.  Each row is one (asset, modality, dim).
--
-- We use three separate vector columns so pgvector can build typed indexes.
-- Only one of the three columns is populated per row (based on embedding_dim).

CREATE TABLE IF NOT EXISTS embeddings (
    id              SERIAL PRIMARY KEY,
    asset_id        INT REFERENCES assets(id) ON DELETE CASCADE,
    modality        VARCHAR(10) NOT NULL,   -- 'text', 'image', 'audio', 'video'
    embedding_dim   INT NOT NULL DEFAULT 768,  -- 768, 1536, or 3072
    embedding_768   vector(768),
    embedding_1536  vector(1536),
    embedding_3072  vector(3072),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Migration: add new columns if upgrading from the old single-column schema
DO $$
BEGIN
    -- Add embedding_dim column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'embeddings' AND column_name = 'embedding_dim'
    ) THEN
        ALTER TABLE embeddings ADD COLUMN embedding_dim INT NOT NULL DEFAULT 768;
    END IF;

    -- Add embedding_768 column (migrate data from old 'embedding' column)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'embeddings' AND column_name = 'embedding_768'
    ) THEN
        ALTER TABLE embeddings ADD COLUMN embedding_768 vector(768);
        -- Copy existing data from old column if it exists
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'embeddings' AND column_name = 'embedding'
        ) THEN
            UPDATE embeddings SET embedding_768 = embedding WHERE embedding IS NOT NULL;
        END IF;
    END IF;

    -- Add embedding_1536 column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'embeddings' AND column_name = 'embedding_1536'
    ) THEN
        ALTER TABLE embeddings ADD COLUMN embedding_1536 vector(1536);
    END IF;

    -- Add embedding_3072 column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'embeddings' AND column_name = 'embedding_3072'
    ) THEN
        ALTER TABLE embeddings ADD COLUMN embedding_3072 vector(3072);
    END IF;
END $$;

-- Indexes for each dimension (ivfflat for fast cosine similarity search)
CREATE INDEX IF NOT EXISTS idx_emb_768 ON embeddings
    USING ivfflat (embedding_768 vector_cosine_ops) WITH (lists = 10);

CREATE INDEX IF NOT EXISTS idx_emb_1536 ON embeddings
    USING ivfflat (embedding_1536 vector_cosine_ops) WITH (lists = 10);

-- pgvector can't index `vector` columns above 2000 dims (an index tuple must
-- fit in one 8 KB page), so index 3072 as halfvec (2-byte floats, max 4000).
-- Queries must use the same cast to hit this index (see search.py).
CREATE INDEX IF NOT EXISTS idx_emb_3072 ON embeddings
    USING hnsw ((embedding_3072::halfvec(3072)) halfvec_cosine_ops);

-- Composite index for fast filtering by (modality, dimension)
CREATE INDEX IF NOT EXISTS idx_emb_modality_dim ON embeddings (modality, embedding_dim);

-- ── Japanese study: dialogue lines, words, review ───────────────────────
-- One row per manga speech bubble or per spoken line in a video chunk.
-- For video, t_start/t_end are absolute seconds in the episode
-- (chunk_start + offset within the chunk).
CREATE TABLE IF NOT EXISTS dialogue_lines (
    id              SERIAL PRIMARY KEY,
    asset_id        INT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    line_index      INT NOT NULL,           -- reading order within the asset
    source          VARCHAR(10) NOT NULL,   -- 'manga' or 'video'
    text_ja         TEXT NOT NULL,
    translation_en  TEXT,
    grammar_note    TEXT,
    bbox            JSONB,                  -- manga bubble box [ymin, xmin, ymax, xmax], 0-1000
    t_start         REAL,
    t_end           REAL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (asset_id, line_index)
);

-- Content words per line (dictionary form), for word search and
-- "hear it in the anime" links.
CREATE TABLE IF NOT EXISTS line_words (
    id          SERIAL PRIMARY KEY,
    line_id     INT NOT NULL REFERENCES dialogue_lines(id) ON DELETE CASCADE,
    lemma       TEXT NOT NULL,
    surface     TEXT NOT NULL,
    reading     TEXT,
    jlpt        SMALLINT                    -- 5 (easiest) … 1, NULL if not listed
);
CREATE INDEX IF NOT EXISTS idx_line_words_lemma ON line_words (lemma);
CREATE INDEX IF NOT EXISTS idx_line_words_line ON line_words (line_id);

-- Spaced-repetition review cards (SM-2), one per word.
CREATE TABLE IF NOT EXISTS review_items (
    id               SERIAL PRIMARY KEY,
    lemma            TEXT NOT NULL UNIQUE,
    reading          TEXT,
    meaning          TEXT,
    jlpt             SMALLINT,
    example_line_id  INT REFERENCES dialogue_lines(id) ON DELETE SET NULL,
    ease             REAL NOT NULL DEFAULT 2.5,
    interval_days    REAL NOT NULL DEFAULT 0,
    reps             INT NOT NULL DEFAULT 0,
    lapses           INT NOT NULL DEFAULT 0,
    due_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_review_due ON review_items (due_at);
