"""
Gemini generateContent client (Vertex AI REST) for structured JSON output.

Used to extract manga speech bubbles from page images and to transcribe
Japanese dialogue from video chunks.
"""

import base64
import json
import threading

from backend.config import GEMINI_MODEL, GEMINI_LOCATION
from backend.services.vertex import model_url, post_with_retry

# Limit concurrent generateContent calls (same reasoning as embedding.py).
GEMINI_MAX_CONCURRENT = 4
_gemini_semaphore = threading.Semaphore(GEMINI_MAX_CONCURRENT)


def generate_json(prompt: str, media: list[tuple[bytes, str]], schema: dict) -> dict | list:
    """
    Send a prompt plus inline media to Gemini and return the parsed JSON reply.

    Args:
        prompt: Instruction text.
        media: List of (bytes, mime_type) parts, e.g. [(png_bytes, "image/png")].
        schema: OpenAPI-style response schema the reply must follow.
    """
    parts = [
        {"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode("utf-8")}}
        for data, mime in media
    ]
    parts.append({"text": prompt})

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    url = model_url(GEMINI_MODEL, "generateContent", location=GEMINI_LOCATION)
    data = post_with_retry(url, payload, _gemini_semaphore, timeout=180)

    candidate = data["candidates"][0]
    text = "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", []))
    if not text:
        raise RuntimeError(f"Gemini returned no text (finishReason={candidate.get('finishReason')})")
    return json.loads(text)
