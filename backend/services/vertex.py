"""
Shared Vertex AI REST helpers: credentials, endpoint URLs, and a POST with
retry/backoff. Used by both the embedding client and the Gemini client.

NOTE: google-auth is lazy-imported to avoid blocking app startup.
"""

import logging
import threading
import time

from backend.config import (
    GOOGLE_CLOUD_PROJECT,
    GOOGLE_CLOUD_LOCATION,
    get_gcp_credentials,
)

# Retry settings for transient errors (connection reset, rate limits, etc.)
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0  # seconds; actual delay = base * 2^attempt (2, 4, 8, 16)

logger = logging.getLogger(__name__)

# Lazy initialization
_credentials = None
_init_lock = threading.Lock()


def _ensure_init():
    """Lazy-initialize credentials."""
    global _credentials
    if _credentials is not None:
        return

    with _init_lock:
        if _credentials is not None:
            return

        from google.oauth2 import service_account
        from google.auth.transport.requests import Request

        creds_dict = get_gcp_credentials()
        if creds_dict:
            creds = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            import google.auth
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

        creds.refresh(Request())
        _credentials = creds


def get_auth_headers() -> dict:
    """Get authenticated headers, refreshing token if needed."""
    _ensure_init()

    from google.auth.transport.requests import Request

    if not _credentials.valid:
        _credentials.refresh(Request())

    return {
        "Authorization": f"Bearer {_credentials.token}",
        "Content-Type": "application/json",
    }


def model_url(model_id: str, method: str, location: str = GOOGLE_CLOUD_LOCATION) -> str:
    """
    Build a publisher-model endpoint URL, e.g. ...models/{model_id}:embedContent.

    The 'global' location has no regional host prefix
    (aiplatform.googleapis.com, not global-aiplatform.googleapis.com).
    """
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return (
        f"https://{host}/v1/"
        f"projects/{GOOGLE_CLOUD_PROJECT}/locations/{location}/"
        f"publishers/google/models/{model_id}:{method}"
    )


def post_with_retry(url: str, payload: dict, semaphore: threading.Semaphore, timeout: int = 60) -> dict:
    """
    POST a JSON payload to a Vertex endpoint and return the JSON response.
    Retries on transient errors (connection reset, 429, 500, 502, 503) with
    exponential backoff.

    The semaphore limits concurrent outbound calls (prevents
    "Connection reset by peer" under high concurrency).
    """
    import requests as req

    # Throttle: wait for a semaphore slot before making the HTTP call
    with semaphore:
        last_exc = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                # Re-fetch headers each attempt (token may have expired)
                headers = get_auth_headers()
                response = req.post(url, headers=headers, json=payload, timeout=timeout)

                # Retry on rate-limit or server errors
                if response.status_code in (429, 500, 502, 503) and attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Vertex returned %d, retrying in %.1fs (attempt %d/%d)",
                        response.status_code, delay, attempt + 1, MAX_RETRIES,
                    )
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                return response.json()

            except (req.exceptions.ConnectionError, ConnectionResetError) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Vertex connection error: %s — retrying in %.1fs (attempt %d/%d)",
                        exc, delay, attempt + 1, MAX_RETRIES,
                    )
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(f"Vertex request failed after {MAX_RETRIES} retries: {last_exc}")
