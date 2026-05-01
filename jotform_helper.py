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
MIN_SUGGESTION_SCORE = 1


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


def _fuzzy_score(needle, haystack):
    """Score how well `needle` matches `haystack`. Higher is better.

    0     = no match
    >=10  = substring match (+ length-based bonus)
    1-9   = token-overlap match only
    """
    if not needle or not haystack:
        return 0
    n = needle.lower().strip()
    h = haystack.lower().strip()

    if n in h:
        return 10 + len(n)

    needle_tokens = set(t.strip(".,!?") for t in n.split() if len(t) > 2)
    haystack_tokens = set(t.strip(".,!?") for t in h.split() if len(t) > 2)
    if not needle_tokens:
        return 0

    matches = 0
    for nt in needle_tokens:
        for ht in haystack_tokens:
            if nt == ht or ht.startswith(nt) or nt.startswith(ht):
                matches += 1
                break
    return matches


def _suggest_form(name):
    """Return the single highest-scoring non-DELETED form, or None if no form scores >= MIN_SUGGESTION_SCORE."""
    needle = (name or "").strip()
    if not needle:
        return None

    forms = _api_get("/user/forms", {"limit": 1000})
    active = [f for f in forms if f.get("status") != "DELETED"]

    best_form = None
    best_score = 0
    for f in active:
        score = _fuzzy_score(needle, f.get("title", ""))
        if score > best_score:
            best_score = score
            best_form = f

    if best_score >= MIN_SUGGESTION_SCORE:
        return best_form
    return None


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
