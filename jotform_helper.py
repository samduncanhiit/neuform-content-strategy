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
    raise NotImplementedError


def get_submission_count(name):
    raise NotImplementedError
