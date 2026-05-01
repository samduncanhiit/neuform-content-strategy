"""
JotForm API helper for WhatsApp bot integration.
Returns submission counts for forms looked up by name.
"""

import os
import logging

import requests

logger = logging.getLogger(__name__)

JOTFORM_BASE = "https://api.jotform.com"
JOTFORM_API_KEY = os.environ.get("JOTFORM_API_KEY")


def _api_get(path, params=None):
    """GET https://api.jotform.com{path}; returns the 'content' field of the JSON response."""
    if not JOTFORM_API_KEY:
        raise RuntimeError("JOTFORM_API_KEY env var is not set")
    url = f"{JOTFORM_BASE}{path}"
    all_params = {"apiKey": JOTFORM_API_KEY}
    if params:
        all_params.update(params)
    resp = requests.get(url, params=all_params, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    return body.get("content", [])


def _find_forms_by_name(name):
    """
    Return a list of form dicts whose title matches `name`.
    - Exact case-insensitive match takes priority and short-circuits.
    - Otherwise return all forms whose title contains `name` (case-insensitive).
    - DELETED-status forms are always excluded.
    """
    needle = (name or "").strip().lower()
    if not needle:
        return []

    forms = _api_get("/user/forms", {"limit": 1000})
    active = [f for f in forms if f.get("status") != "DELETED"]

    exact = [f for f in active if f.get("title", "").strip().lower() == needle]
    if exact:
        return exact

    return [f for f in active if needle in f.get("title", "").lower()]


def get_submission_count(name):
    raise NotImplementedError
