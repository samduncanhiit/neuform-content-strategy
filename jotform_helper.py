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
    """
    Look up a form by name and return its total submission count.

    Returns a dict with one of these shapes:
      {"status": "ok", "title": str, "count": int}
      {"status": "none", "name": str}
      {"status": "multiple", "matches": [{"title": str, "count": int}, ...]}
      {"status": "error", "message": str}
    """
    try:
        matches = _find_forms_by_name(name)
    except RuntimeError as e:
        # Missing API key
        logger.warning("JotForm not configured: %s", e)
        return {"status": "error", "message": "JotForm is not configured."}
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 401:
            return {
                "status": "error",
                "message": "JotForm API key is invalid — check the Railway env var.",
            }
        logger.exception("JotForm HTTP error")
        return {
            "status": "error",
            "message": "Couldn't reach JotForm right now — try again in a moment.",
        }
    except requests.RequestException:
        logger.exception("JotForm network error")
        return {
            "status": "error",
            "message": "Couldn't reach JotForm right now — try again in a moment.",
        }

    if not matches:
        return {"status": "none", "name": name}

    if len(matches) == 1:
        f = matches[0]
        return {
            "status": "ok",
            "title": f.get("title", ""),
            "count": int(f.get("count", 0) or 0),
        }

    return {
        "status": "multiple",
        "matches": [
            {"title": f.get("title", ""), "count": int(f.get("count", 0) or 0)}
            for f in matches
        ],
    }
