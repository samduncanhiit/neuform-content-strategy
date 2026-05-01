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
    raise NotImplementedError


def _find_forms_by_name(name):
    raise NotImplementedError


def get_submission_count(name):
    raise NotImplementedError
