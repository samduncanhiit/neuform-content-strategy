"""
Trello API helper for WhatsApp bot integration.
Reads cards from the HIIT Challenge board that are due today or overdue.
"""

import os
import logging
from datetime import datetime, timedelta, timezone

import requests

# AEST timezone (UTC+10)
AEST = timezone(timedelta(hours=10))

logger = logging.getLogger(__name__)

TRELLO_BASE = "https://api.trello.com/1"
TRELLO_API_KEY = os.environ.get("TRELLO_API_KEY")
TRELLO_TOKEN = os.environ.get("TRELLO_TOKEN")

# Cache for board ID lookup
_board_id_cache = None
_board_id_cache_by_name = {}
_list_cache_by_board = {}  # board_id -> (timestamp, [list dicts])
_LIST_CACHE_TTL_SEC = 600  # 10 minutes


def _now():
    """Get current time in AEST."""
    return datetime.now(AEST)


def _trello_params(extra=None):
    """Build base query params with auth credentials."""
    params = {
        "key": TRELLO_API_KEY,
        "token": TRELLO_TOKEN,
    }
    if extra:
        params.update(extra)
    return params


def _trello_get(path, params=None):
    """Make a GET request to the Trello API."""
    url = f"{TRELLO_BASE}/{path}"
    all_params = _trello_params(params)
    resp = requests.get(url, params=all_params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _trello_post(path, params=None, json_body=None):
    """Make a POST request to the Trello API."""
    url = f"{TRELLO_BASE}/{path}"
    all_params = _trello_params(params)
    resp = requests.post(url, params=all_params, json=json_body, timeout=60)
    resp.raise_for_status()
    return resp.json() if resp.text else {}


def _trello_put(path, params=None, json_body=None):
    """Make a PUT request to the Trello API."""
    url = f"{TRELLO_BASE}/{path}"
    all_params = _trello_params(params)
    resp = requests.put(url, params=all_params, json=json_body, timeout=60)
    resp.raise_for_status()
    return resp.json() if resp.text else {}


def _find_board_id(board_name):
    """Look up a board ID by case-insensitive name. Cached per-name forever."""
    if not board_name:
        return None
    key = board_name.lower().strip()
    if key in _board_id_cache_by_name:
        return _board_id_cache_by_name[key]

    boards = _trello_get("members/me/boards", {"fields": "name,id"})
    for board in boards:
        if (board.get("name") or "").lower().strip() == key:
            _board_id_cache_by_name[key] = board["id"]
            return board["id"]
    return None


def _find_list(board_id, list_name):
    """Find a list on a board by exact (case-insensitive) name.

    Returns (list_id, canonical_name) or None.
    Lists are cached per board for _LIST_CACHE_TTL_SEC seconds.
    """
    if not board_id or not list_name:
        return None

    import time
    now = time.time()
    cached = _list_cache_by_board.get(board_id)
    if cached and (now - cached[0]) < _LIST_CACHE_TTL_SEC:
        lists = cached[1]
    else:
        lists = _trello_get(f"boards/{board_id}/lists", {"fields": "name,id"})
        _list_cache_by_board[board_id] = (now, lists)

    key = list_name.lower().strip()
    for lst in lists:
        if (lst.get("name") or "").lower().strip() == key:
            return (lst["id"], lst["name"])
    return None


def _find_cards(board_id, title):
    raise NotImplementedError


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


def _resolve_labels(board_id, label_names):
    raise NotImplementedError


def create_card(board_id, list_id, title, due_date=None, description=None, label_ids=None):
    raise NotImplementedError


def update_card(card_id, **fields):
    raise NotImplementedError


def move_card(card_id, dest_list_id):
    raise NotImplementedError


def archive_card(card_id):
    raise NotImplementedError


def _find_hiit_challenge_board():
    """Find the HIIT Challenge board ID by searching user's boards."""
    global _board_id_cache
    if _board_id_cache:
        return _board_id_cache

    boards = _trello_get("members/me/boards", {"fields": "name,id"})
    for board in boards:
        name = (board.get("name") or "").lower()
        if "hiit" in name and "challenge" in name:
            _board_id_cache = board["id"]
            logger.info(f"Found HIIT Challenge board: {board['name']} ({board['id']})")
            return _board_id_cache

    # If no exact match, return first board with "hiit" in the name
    for board in boards:
        name = (board.get("name") or "").lower()
        if "hiit" in name:
            _board_id_cache = board["id"]
            logger.info(f"Using board: {board['name']} ({board['id']})")
            return _board_id_cache

    logger.warning("HIIT Challenge board not found")
    return None


def get_trello_tasks():
    """Get cards from the HIIT Challenge board that are due today or overdue.

    Returns list of cards with name, due date, list name, and overdue status.
    """
    board_id = _find_hiit_challenge_board()
    if not board_id:
        return {"error": "HIIT Challenge board not found", "tasks": []}

    # Get all open cards with due dates
    cards = _trello_get(
        f"boards/{board_id}/cards",
        {
            "fields": "name,due,dueComplete,idList",
            "filter": "open",
        },
    )

    # Get lists for mapping list IDs to names
    lists = _trello_get(f"boards/{board_id}/lists", {"fields": "name,id"})
    list_map = {lst["id"]: lst["name"] for lst in lists}

    now = _now()
    today_date = now.strftime("%Y-%m-%d")

    tasks = []
    for card in cards:
        due = card.get("due")
        if not due:
            continue
        if card.get("dueComplete"):
            continue

        # Parse due date
        try:
            due_dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
            due_aest = due_dt.astimezone(AEST)
        except (ValueError, TypeError):
            continue

        due_date_str = due_aest.strftime("%Y-%m-%d")
        is_overdue = due_aest < now
        is_today = due_date_str == today_date

        if is_today or is_overdue:
            tasks.append({
                "name": card.get("name", "Untitled"),
                "due_date": due_aest.strftime("%a %-d %b %-I:%M %p"),
                "list": list_map.get(card.get("idList"), "Unknown"),
                "overdue": is_overdue and not is_today,
            })

    # Sort: overdue first, then by name
    tasks.sort(key=lambda x: (not x["overdue"], x["name"]))

    return {
        "board_name": "HIIT Challenge",
        "total_tasks": len(tasks),
        "tasks": tasks,
    }


def format_trello_tasks(data):
    """Format Trello tasks for WhatsApp."""
    if data.get("error"):
        return f"*Trello*\n{data['error']}"

    tasks = data.get("tasks", [])
    if not tasks:
        return "*HIIT Challenge Tasks*\n\nNo tasks due today or overdue."

    overdue = [t for t in tasks if t["overdue"]]
    today = [t for t in tasks if not t["overdue"]]

    lines = [f"*HIIT Challenge Tasks*\n"]

    if overdue:
        lines.append(f"*OVERDUE ({len(overdue)})*")
        for t in overdue:
            lines.append(f"  - {t['name']}\n    Due: {t['due_date']}\n    List: {t['list']}")
        lines.append("")

    if today:
        lines.append(f"*DUE TODAY ({len(today)})*")
        for t in today:
            lines.append(f"  - {t['name']}\n    Due: {t['due_date']}\n    List: {t['list']}")

    lines.append(f"\nTotal: {len(tasks)} task{'s' if len(tasks) != 1 else ''}")
    return "\n".join(lines)
