"""
Vector search service: queries PostgreSQL (pgvector) for nearest neighbor embeddings.
Supports multiple embedding dimensions (768, 1536, 3072).
"""

from backend.db import get_db
from backend.services.embedding import (
    embed_text,
    embed_image,
    embed_audio,
    DEFAULT_EMBEDDING_DIM,
    ALLOWED_EMBEDDING_DIMS,
)


def _embedding_column(dim: int) -> str:
    """Return the DB column name for the given embedding dimension."""
    if dim not in ALLOWED_EMBEDDING_DIMS:
        raise ValueError(f"Invalid embedding dimension {dim}. Allowed: {ALLOWED_EMBEDDING_DIMS}")
    return f"embedding_{dim}"


def _vector_search(
    query_embedding: list[float],
    target_modality: str,
    limit: int = 10,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> list[dict]:
    """
    Perform vector similarity search in PostgreSQL (pgvector).
    
    Args:
        query_embedding: The query vector
        target_modality: Filter results by modality ('image', 'audio', 'video')
        limit: Max number of results
        embedding_dim: Which embedding dimension to search (768, 1536, or 3072)
    
    Returns:
        List of matching assets with distance scores
    """
    col = _embedding_column(embedding_dim)
    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    # 3072 is indexed as halfvec (pgvector can't index vector > 2000 dims),
    # so compare as halfvec on both sides to use idx_emb_3072.
    if embedding_dim == 3072:
        col_expr = f"e.{col}::halfvec(3072)"
        query_cast = "halfvec(3072)"
    else:
        col_expr = f"e.{col}"
        query_cast = "vector"

    sql = f"""
        SELECT
            a.id AS asset_id,
            a.file_path,
            a.asset_type,
            a.dialogue_text,
            a.translation_en,
            a.manga_id,
            a.page_number,
            a.chunk_start,
            a.chunk_end,
            m.title AS manga_title,
            {col_expr} <=> %s::{query_cast} AS distance
        FROM embeddings e
        JOIN assets a ON e.asset_id = a.id
        LEFT JOIN manga m ON a.manga_id = m.id
        WHERE e.modality = %s
          AND e.embedding_dim = %s
          AND e.{col} IS NOT NULL
        ORDER BY {col_expr} <=> %s::{query_cast}
        LIMIT %s;
    """

    with get_db() as conn:
        cursor = conn.cursor()
        # Set ivfflat.probes high enough to get accurate results.
        # With lists=10, probes=10 means exhaustive search across all lists,
        # guaranteeing exact results regardless of LIMIT value.
        cursor.execute("SET ivfflat.probes = 10")
        # The 3072 HNSW index returns ef_search candidates *before* the
        # modality filter is applied; widen it so filtered searches still
        # fill LIMIT.
        cursor.execute("SET hnsw.ef_search = 200")
        cursor.execute(sql, (embedding_str, target_modality, embedding_dim, embedding_str, limit))
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()

    return [dict(zip(columns, row)) for row in rows]


def text_to_image(query: str, limit: int = 10, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> list[dict]:
    """Search for images using a text query."""
    embedding = embed_text(query, dim=embedding_dim)
    return _vector_search(embedding, target_modality="image", limit=limit, embedding_dim=embedding_dim)


def text_to_audio(query: str, limit: int = 10, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> list[dict]:
    """Search for audio using a text query."""
    embedding = embed_text(query, dim=embedding_dim)
    return _vector_search(embedding, target_modality="audio", limit=limit, embedding_dim=embedding_dim)


def text_to_video(query: str, limit: int = 10, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> list[dict]:
    """Search for video using a text query."""
    embedding = embed_text(query, dim=embedding_dim)
    return _vector_search(embedding, target_modality="video", limit=limit, embedding_dim=embedding_dim)


def image_to_image(image_bytes: bytes, limit: int = 10, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> list[dict]:
    """Search for similar images using an image."""
    embedding = embed_image(image_bytes, dim=embedding_dim)
    return _vector_search(embedding, target_modality="image", limit=limit, embedding_dim=embedding_dim)


def image_to_video(image_bytes: bytes, limit: int = 10, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> list[dict]:
    """Search for related videos using an image."""
    embedding = embed_image(image_bytes, dim=embedding_dim)
    return _vector_search(embedding, target_modality="video", limit=limit, embedding_dim=embedding_dim)


def audio_to_video(
    audio_bytes: bytes,
    limit: int = 10,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    mime_type: str = "audio/mpeg",
) -> list[dict]:
    """Search video scenes with a short audio clip (dialogue, music, sound effects)."""
    embedding = embed_audio(audio_bytes, dim=embedding_dim, mime_type=mime_type)
    return audio_embedding_to_video(embedding, limit, embedding_dim)


def audio_embedding_to_video(
    embedding: list[float],
    limit: int = 10,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> list[dict]:
    """
    Match an audio embedding against the chunks' *audio tracks*, then return
    the video chunks covering the same time range.

    Comparing audio with video embeddings fails for music-heavy scenes: a
    video embedding is dominated by its visuals (an opening-song clip ranked
    its own chunk #11 against video, #1 against the audio track). Falls back
    to video embeddings when no audio tracks are indexed
    (scripts/index_audio_tracks.py).
    """
    tracks = _vector_search(embedding, target_modality="audio", limit=limit * 2, embedding_dim=embedding_dim)
    tracks = [t for t in tracks if t["chunk_start"] is not None and t["manga_id"] is not None]
    if not tracks:
        return _vector_search(embedding, target_modality="video", limit=limit, embedding_dim=embedding_dim)

    keys = tuple((t["manga_id"], t["chunk_start"]) for t in tracks)
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT a.id AS asset_id, a.file_path, a.asset_type, a.dialogue_text,
                   a.translation_en, a.manga_id, a.page_number, a.chunk_start,
                   a.chunk_end, m.title AS manga_title
            FROM assets a LEFT JOIN manga m ON a.manga_id = m.id
            WHERE a.asset_type = 'video' AND (a.manga_id, a.chunk_start) IN %s
            """,
            (keys,),
        )
        columns = [desc[0] for desc in cursor.description]
        videos = {(r["manga_id"], r["chunk_start"]): r for r in (dict(zip(columns, row)) for row in cursor.fetchall())}
        cursor.close()

    results = []
    for t in tracks:
        video = videos.get((t["manga_id"], t["chunk_start"]))
        if video and video not in results:
            results.append({**video, "distance": t["distance"]})
        if len(results) == limit:
            break
    return results


def text_to_pdf(query: str, limit: int = 10, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> list[dict]:
    """Search for PDF document pages using a text query."""
    embedding = embed_text(query, dim=embedding_dim)
    return _vector_search(embedding, target_modality="pdf", limit=limit, embedding_dim=embedding_dim)


def asset_to_video(asset_id: int, limit: int = 3, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> list[dict]:
    """
    Find video scenes for an already-ingested asset (e.g. a manga page)
    using its stored embedding — no new embedding API call.
    """
    col = _embedding_column(embedding_dim)
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {col}::text FROM embeddings WHERE asset_id = %s AND embedding_dim = %s LIMIT 1",
            (asset_id, embedding_dim),
        )
        row = cursor.fetchone()
        cursor.close()
    if row is None or row[0] is None:
        return []
    embedding = [float(v) for v in row[0].strip("[]").split(",")]
    return _vector_search(embedding, target_modality="video", limit=limit, embedding_dim=embedding_dim)
