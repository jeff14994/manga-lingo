"""
Gemini Embedding 2 Preview client for multimodal embeddings.
Uses the gemini-embedding-2-preview model which supports text, image,
audio, and video inputs in the same embedding space.

Uses the :embedContent REST endpoint on Vertex AI.

NOTE: Vertex AI SDK is lazy-imported to avoid blocking app startup.
"""

import base64
import io
import logging
import mimetypes
import threading

from backend.services.vertex import model_url, post_with_retry

MODEL_ID = "gemini-embedding-2-preview"
DEFAULT_EMBEDDING_DIM = 768
ALLOWED_EMBEDDING_DIMS = (768, 1536, 3072)

# ── Concurrency Throttle ────────────────────────────────────────────────
# Limit concurrent outbound calls to the Gemini embedContent API.
# When the FastAPI server handles multiple ingest requests simultaneously
# (e.g. batch_ingest.py with --concurrency 5), all of them call this module.
# Without throttling, 5+ simultaneous HTTPS requests to the same endpoint
# overwhelm the connection and cause "Connection reset by peer".
#
# This semaphore ensures at most EMBED_MAX_CONCURRENT requests are in-flight
# at any given time, regardless of how many FastAPI worker threads are active.
EMBED_MAX_CONCURRENT = 3
_embed_semaphore = threading.Semaphore(EMBED_MAX_CONCURRENT)

logger = logging.getLogger(__name__)

# Supported image MIME types for the embedContent endpoint.
# WebP is NOT supported — images will be auto-converted to JPEG.
_SUPPORTED_IMAGE_MIMES = {"image/png", "image/jpeg", "image/gif", "image/bmp", "image/tiff"}


def _call_embed_content(parts: list[dict], dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """
    Call the Vertex AI embedContent endpoint with the given parts.
    Retries and throttling are handled by vertex.post_with_retry.

    Args:
        parts: Content parts to embed.
        dim: Output embedding dimension (768, 1536, or 3072).

    Returns the embedding vector.
    """
    if dim not in ALLOWED_EMBEDDING_DIMS:
        raise ValueError(f"Invalid embedding dimension {dim}. Allowed: {ALLOWED_EMBEDDING_DIMS}")

    payload = {
        "content": {
            "parts": parts,
        },
        "outputDimensionality": dim,
    }
    data = post_with_retry(model_url(MODEL_ID, "embedContent"), payload, _embed_semaphore)
    return data["embedding"]["values"]


def _convert_image_to_jpeg(image_bytes: bytes) -> tuple[bytes, str]:
    """
    Convert image bytes to JPEG if the format is not natively supported
    (e.g. WebP). Returns (converted_bytes, mime_type).
    """
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    fmt = (img.format or "").upper()

    # If already a supported format, detect and return as-is
    if fmt == "PNG":
        return image_bytes, "image/png"
    elif fmt in ("JPEG", "JPG"):
        return image_bytes, "image/jpeg"
    elif fmt == "GIF":
        return image_bytes, "image/gif"

    # For unsupported formats (WebP, BMP, TIFF, etc.), convert to JPEG
    # Ensure RGB mode (JPEG doesn't support RGBA)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


# ── Public API ───────────────────────────────────────────────────────────
# All public functions accept an optional `dim` parameter (768, 1536, 3072).
# When omitted, DEFAULT_EMBEDDING_DIM (768) is used.


def embed_text(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """Generate embedding for a text query."""
    return _call_embed_content([{"text": text}], dim=dim)


def embed_image(image_bytes: bytes, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """
    Generate embedding for an image.
    Automatically converts unsupported formats (e.g. WebP) to JPEG.
    """
    converted_bytes, mime_type = _convert_image_to_jpeg(image_bytes)
    b64_data = base64.b64encode(converted_bytes).decode("utf-8")

    return _call_embed_content([
        {
            "inlineData": {
                "mimeType": mime_type,
                "data": b64_data,
            }
        }
    ], dim=dim)


def embed_image_from_path(file_path: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """Generate embedding for an image from a local file path."""
    with open(file_path, "rb") as f:
        return embed_image(f.read(), dim=dim)


def embed_video(video_bytes: bytes, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """Generate embedding for a video."""
    b64_data = base64.b64encode(video_bytes).decode("utf-8")

    return _call_embed_content([
        {
            "inlineData": {
                "mimeType": "video/mp4",
                "data": b64_data,
            }
        }
    ], dim=dim)


def embed_video_from_path(file_path: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """Generate embedding for a video from a local file path."""
    with open(file_path, "rb") as f:
        return embed_video(f.read(), dim=dim)


def embed_audio(audio_bytes: bytes, dim: int = DEFAULT_EMBEDDING_DIM, mime_type: str = "audio/mpeg") -> list[float]:
    """Generate embedding for audio (mp3 by default; pass mime_type for wav/m4a/etc.)."""
    b64_data = base64.b64encode(audio_bytes).decode("utf-8")

    return _call_embed_content([
        {
            "inlineData": {
                "mimeType": mime_type,
                "data": b64_data,
            }
        }
    ], dim=dim)


def embed_audio_from_path(file_path: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """Generate embedding for audio from a local file path."""
    mime_type = mimetypes.guess_type(file_path)[0] or "audio/mpeg"
    with open(file_path, "rb") as f:
        return embed_audio(f.read(), dim=dim, mime_type=mime_type)


def embed_document(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """
    Generate embedding for a document page/chunk text.
    Uses the same text embedding as queries — both sides live in the same
    embedding space, so text→document search works via cosine similarity.
    """
    return _call_embed_content([{"text": text}], dim=dim)
