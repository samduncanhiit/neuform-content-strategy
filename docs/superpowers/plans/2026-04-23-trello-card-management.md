# Trello Card Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Trello card add/edit/move/archive tools to the HIIT WhatsApp bot, scoped to Erin's HIIT Office board with default To Do / Done lists.

**Architecture:** New write helpers in `trello_helper.py`, new tool definitions and handlers in `app.py` behind a `USER_TRELLO` gate that filters the tools out for non-configured users. Fuzzy matching for card lookup; two-step confirmation for archive.

**Tech Stack:** Python 3, Flask, Trello REST API v1, `requests`, `unittest` (existing test framework), deploy via `railway up --detach`.

**Spec:** `docs/superpowers/specs/2026-04-23-trello-card-management-design.md`

---

## File Structure

- **Modify** `trello_helper.py` — add write helpers: `_trello_post`, `_trello_put`, `_find_board_id`, `_find_list`, `_find_cards`, `_fuzzy_score`, `_resolve_labels`, `create_card`, `update_card`, `move_card`, `archive_card`.
- **Modify** `app.py` — add `USER_TRELLO` dict, `_TRELLO_WRITE_TOOLS` list, 4 handlers in `handle_tool_call`, thread `raw_number` parameter through, update `_get_tools_for_user`, update `SYSTEM_PROMPT`.
- **Create** `tests/test_trello_write.py` — unit tests for pure logic (`_fuzzy_score`) and mocked tests for API wrappers and handlers.

---

## Task 1: Scaffolding — add `_trello_post` / `_trello_put` + test import

**Files:**
- Modify: `trello_helper.py` (around line 47, after `_trello_get`)
- Create: `tests/test_trello_write.py`

- [ ] **Step 1: Write a scaffolding test that asserts the new helpers exist**

Create `tests/test_trello_write.py`:

```python
"""Unit tests for Trello write helpers and tool handlers."""
import unittest
from unittest.mock import patch, MagicMock

import trello_helper


class TestScaffolding(unittest.TestCase):
    def test_module_exposes_write_helpers(self):
        for name in [
            "_trello_post", "_trello_put",
            "_find_board_id", "_find_list", "_find_cards",
            "_fuzzy_score", "_resolve_labels",
            "create_card", "update_card", "move_card", "archive_card",
        ]:
            self.assertTrue(hasattr(trello_helper, name), f"missing {name}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_trello_write -v`
Expected: FAIL with "missing _trello_post" (or whichever is checked first).

- [ ] **Step 3: Add `_trello_post` and `_trello_put` to `trello_helper.py`**

Insert after the existing `_trello_get` function (around line 47):

```python
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
```

Also add stubs for the rest so the scaffolding test passes — we'll implement them in later tasks:

```python
def _find_board_id(board_name):
    raise NotImplementedError

def _find_list(board_id, list_name):
    raise NotImplementedError

def _find_cards(board_id, title):
    raise NotImplementedError

def _fuzzy_score(needle, haystack):
    raise NotImplementedError

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_trello_write -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add trello_helper.py tests/test_trello_write.py
git commit -m "Scaffold Trello write helpers"
```

---

## Task 2: Implement `_fuzzy_score` (pure function, TDD)

**Files:**
- Modify: `trello_helper.py` (replace the `_fuzzy_score` stub)
- Modify: `tests/test_trello_write.py` (add test class)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_trello_write.py`:

```python
class TestFuzzyScore(unittest.TestCase):
    def test_exact_match_is_highest(self):
        self.assertGreater(
            trello_helper._fuzzy_score("buy kettlebells", "buy kettlebells"),
            trello_helper._fuzzy_score("buy kettlebells", "order dumbbells"),
        )

    def test_substring_match_beats_non_match(self):
        self.assertGreater(
            trello_helper._fuzzy_score("kettlebell", "buy kettlebells for gym"),
            trello_helper._fuzzy_score("kettlebell", "order new mats"),
        )

    def test_case_insensitive(self):
        self.assertEqual(
            trello_helper._fuzzy_score("Kettlebell", "buy kettlebell"),
            trello_helper._fuzzy_score("kettlebell", "BUY KETTLEBELL"),
        )

    def test_token_overlap_scores(self):
        # "kettlebell order" should match "order 4 kettlebells" via tokens
        self.assertGreater(
            trello_helper._fuzzy_score("kettlebell order", "order 4 kettlebells"),
            trello_helper._fuzzy_score("kettlebell order", "pay power bill"),
        )

    def test_empty_needle_returns_zero(self):
        self.assertEqual(trello_helper._fuzzy_score("", "anything"), 0)

    def test_no_overlap_returns_zero(self):
        self.assertEqual(
            trello_helper._fuzzy_score("kettlebell", "pay power bill"),
            0,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trello_write.TestFuzzyScore -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `_fuzzy_score`**

Replace the stub in `trello_helper.py`:

```python
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
        # Longer needle matching = more specific = higher score
        return 10 + len(n)

    needle_tokens = set(t.strip(".,!?") for t in n.split() if len(t) > 2)
    haystack_tokens = set(t.strip(".,!?") for t in h.split() if len(t) > 2)
    if not needle_tokens:
        return 0

    # Count tokens that appear in haystack (exact or as prefix, e.g. kettlebell/kettlebells)
    matches = 0
    for nt in needle_tokens:
        for ht in haystack_tokens:
            if nt == ht or ht.startswith(nt) or nt.startswith(ht):
                matches += 1
                break
    return matches
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_trello_write.TestFuzzyScore -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add trello_helper.py tests/test_trello_write.py
git commit -m "Add _fuzzy_score matcher for Trello card lookup"
```

---

## Task 3: Implement `_find_board_id` and `_find_list`

**Files:**
- Modify: `trello_helper.py`
- Modify: `tests/test_trello_write.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_trello_write.py`:

```python
class TestFindBoard(unittest.TestCase):
    def setUp(self):
        # Reset the module-level caches between tests
        trello_helper._board_id_cache_by_name = {}

    @patch("trello_helper._trello_get")
    def test_finds_board_by_exact_name(self, mock_get):
        mock_get.return_value = [
            {"id": "b1", "name": "HIIT Office"},
            {"id": "b2", "name": "HIIT Challenge"},
        ]
        self.assertEqual(trello_helper._find_board_id("HIIT Office"), "b1")

    @patch("trello_helper._trello_get")
    def test_is_case_insensitive(self, mock_get):
        mock_get.return_value = [{"id": "b1", "name": "HIIT Office"}]
        self.assertEqual(trello_helper._find_board_id("hiit office"), "b1")

    @patch("trello_helper._trello_get")
    def test_returns_none_when_no_match(self, mock_get):
        mock_get.return_value = [{"id": "b1", "name": "Other Board"}]
        self.assertIsNone(trello_helper._find_board_id("HIIT Office"))

    @patch("trello_helper._trello_get")
    def test_caches_result(self, mock_get):
        mock_get.return_value = [{"id": "b1", "name": "HIIT Office"}]
        trello_helper._find_board_id("HIIT Office")
        trello_helper._find_board_id("HIIT Office")
        self.assertEqual(mock_get.call_count, 1)


class TestFindList(unittest.TestCase):
    def setUp(self):
        trello_helper._list_cache_by_board = {}

    @patch("trello_helper._trello_get")
    def test_finds_list_by_exact_name(self, mock_get):
        mock_get.return_value = [
            {"id": "l1", "name": "To Do List"},
            {"id": "l2", "name": "Done"},
        ]
        result = trello_helper._find_list("board1", "Done")
        self.assertEqual(result, ("l2", "Done"))

    @patch("trello_helper._trello_get")
    def test_is_case_insensitive(self, mock_get):
        mock_get.return_value = [{"id": "l1", "name": "To Do List"}]
        result = trello_helper._find_list("board1", "to do list")
        self.assertEqual(result, ("l1", "To Do List"))

    @patch("trello_helper._trello_get")
    def test_returns_none_when_no_match(self, mock_get):
        mock_get.return_value = [{"id": "l1", "name": "Done"}]
        self.assertIsNone(trello_helper._find_list("board1", "Nonexistent"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trello_write.TestFindBoard tests.test_trello_write.TestFindList -v`
Expected: FAIL with `NotImplementedError` / missing attribute.

- [ ] **Step 3: Implement the helpers**

Add to `trello_helper.py` (near the top, below the existing `_board_id_cache`):

```python
_board_id_cache_by_name = {}
_list_cache_by_board = {}  # board_id -> (timestamp, [list dicts])
_LIST_CACHE_TTL_SEC = 600  # 10 minutes
```

Replace the `_find_board_id` stub:

```python
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
```

Replace the `_find_list` stub:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_trello_write.TestFindBoard tests.test_trello_write.TestFindList -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add trello_helper.py tests/test_trello_write.py
git commit -m "Add _find_board_id and _find_list with caching"
```

---

## Task 4: Implement `_find_cards`

**Files:**
- Modify: `trello_helper.py`
- Modify: `tests/test_trello_write.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_trello_write.py`:

```python
class TestFindCards(unittest.TestCase):
    @patch("trello_helper._trello_get")
    def test_returns_sorted_matches(self, mock_get):
        # First call -> board cards; second call -> lists
        mock_get.side_effect = [
            [
                {"id": "c1", "name": "Buy kettlebells", "idList": "l1",
                 "shortUrl": "https://trello.com/c/c1"},
                {"id": "c2", "name": "Order dumbbells", "idList": "l1",
                 "shortUrl": "https://trello.com/c/c2"},
                {"id": "c3", "name": "kettlebell rack install", "idList": "l2",
                 "shortUrl": "https://trello.com/c/c3"},
            ],
            [
                {"id": "l1", "name": "To Do List"},
                {"id": "l2", "name": "Doing"},
            ],
        ]
        matches = trello_helper._find_cards("board1", "kettlebell")
        self.assertEqual(len(matches), 2)
        # Sorted highest score first
        self.assertEqual(matches[0]["name"], "Buy kettlebells")
        self.assertEqual(matches[0]["list_name"], "To Do List")
        self.assertEqual(matches[0]["url"], "https://trello.com/c/c1")

    @patch("trello_helper._trello_get")
    def test_no_matches_returns_empty(self, mock_get):
        mock_get.side_effect = [
            [{"id": "c1", "name": "Pay power bill", "idList": "l1", "shortUrl": ""}],
            [{"id": "l1", "name": "Bills"}],
        ]
        self.assertEqual(trello_helper._find_cards("board1", "kettlebell"), [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trello_write.TestFindCards -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `_find_cards`**

Replace the stub:

```python
def _find_cards(board_id, title):
    """Find cards on a board by fuzzy title match.

    Returns a list of dicts (highest-score first):
        [{"id": ..., "name": ..., "list_id": ..., "list_name": ..., "url": ..., "score": ...}]
    """
    if not board_id or not title:
        return []

    cards = _trello_get(
        f"boards/{board_id}/cards",
        {"fields": "name,idList,shortUrl", "filter": "open"},
    )
    lists = _trello_get(f"boards/{board_id}/lists", {"fields": "name,id"})
    list_name_by_id = {lst["id"]: lst["name"] for lst in lists}

    results = []
    for c in cards:
        score = _fuzzy_score(title, c.get("name") or "")
        if score > 0:
            results.append({
                "id": c["id"],
                "name": c.get("name") or "",
                "list_id": c.get("idList"),
                "list_name": list_name_by_id.get(c.get("idList"), "Unknown"),
                "url": c.get("shortUrl") or "",
                "score": score,
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_trello_write.TestFindCards -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add trello_helper.py tests/test_trello_write.py
git commit -m "Add _find_cards with fuzzy-matched, score-sorted results"
```

---

## Task 5: Implement `_resolve_labels`

**Files:**
- Modify: `trello_helper.py`
- Modify: `tests/test_trello_write.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_trello_write.py`:

```python
class TestResolveLabels(unittest.TestCase):
    @patch("trello_helper._trello_post")
    @patch("trello_helper._trello_get")
    def test_returns_existing_label_ids_and_creates_missing(self, mock_get, mock_post):
        mock_get.return_value = [
            {"id": "lab1", "name": "Urgent"},
            {"id": "lab2", "name": "Admin"},
        ]
        mock_post.return_value = {"id": "lab3", "name": "New"}

        ids = trello_helper._resolve_labels("board1", ["Urgent", "New"])

        self.assertEqual(ids, ["lab1", "lab3"])
        mock_post.assert_called_once()
        # Verify it was called with the right path and name
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "boards/board1/labels")
        self.assertEqual(kwargs.get("params", {}).get("name"), "New")

    @patch("trello_helper._trello_get")
    def test_empty_list_returns_empty(self, mock_get):
        self.assertEqual(trello_helper._resolve_labels("board1", []), [])
        mock_get.assert_not_called()

    @patch("trello_helper._trello_post")
    @patch("trello_helper._trello_get")
    def test_case_insensitive_match(self, mock_get, mock_post):
        mock_get.return_value = [{"id": "lab1", "name": "Urgent"}]
        ids = trello_helper._resolve_labels("board1", ["urgent"])
        self.assertEqual(ids, ["lab1"])
        mock_post.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trello_write.TestResolveLabels -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `_resolve_labels`**

Replace the stub:

```python
def _resolve_labels(board_id, label_names):
    """Turn a list of label names into label IDs. Creates missing labels on the board.

    Label matching is case-insensitive.
    """
    if not label_names:
        return []

    existing = _trello_get(f"boards/{board_id}/labels", {"fields": "name,id"})
    by_lower = {(l.get("name") or "").lower().strip(): l["id"] for l in existing}

    ids = []
    for name in label_names:
        key = name.lower().strip()
        if key in by_lower:
            ids.append(by_lower[key])
        else:
            created = _trello_post(
                f"boards/{board_id}/labels",
                params={"name": name, "color": ""},
            )
            ids.append(created["id"])
            by_lower[key] = created["id"]
    return ids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_trello_write.TestResolveLabels -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add trello_helper.py tests/test_trello_write.py
git commit -m "Add _resolve_labels — auto-creates missing labels on board"
```

---

## Task 6: Implement `create_card`, `update_card`, `move_card`, `archive_card`

**Files:**
- Modify: `trello_helper.py`
- Modify: `tests/test_trello_write.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_trello_write.py`:

```python
class TestCardWriteWrappers(unittest.TestCase):
    @patch("trello_helper._trello_post")
    def test_create_card_sends_required_fields(self, mock_post):
        mock_post.return_value = {"id": "c1", "shortUrl": "https://trello.com/c/c1"}
        result = trello_helper.create_card(
            board_id="b1", list_id="l1", title="Buy kettlebells",
            due_date="2026-04-25", description="From supplier X",
            label_ids=["lab1", "lab2"],
        )
        self.assertEqual(result["id"], "c1")
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "cards")
        params = kwargs.get("params", {})
        self.assertEqual(params["idList"], "l1")
        self.assertEqual(params["name"], "Buy kettlebells")
        self.assertEqual(params["due"], "2026-04-25T10:00:00.000Z")
        self.assertEqual(params["desc"], "From supplier X")
        self.assertEqual(params["idLabels"], "lab1,lab2")

    @patch("trello_helper._trello_post")
    def test_create_card_without_optional_fields(self, mock_post):
        mock_post.return_value = {"id": "c1", "shortUrl": ""}
        trello_helper.create_card(board_id="b1", list_id="l1", title="t")
        args, kwargs = mock_post.call_args
        params = kwargs.get("params", {})
        self.assertEqual(params["name"], "t")
        self.assertNotIn("due", params)
        self.assertNotIn("desc", params)
        self.assertNotIn("idLabels", params)

    @patch("trello_helper._trello_put")
    def test_update_card_passes_through_fields(self, mock_put):
        mock_put.return_value = {"id": "c1"}
        trello_helper.update_card(
            "c1", name="New title", due_date="2026-05-01",
            description="New desc", label_ids=["lab1"],
        )
        args, kwargs = mock_put.call_args
        self.assertEqual(args[0], "cards/c1")
        params = kwargs.get("params", {})
        self.assertEqual(params["name"], "New title")
        self.assertEqual(params["due"], "2026-05-01T10:00:00.000Z")
        self.assertEqual(params["desc"], "New desc")
        self.assertEqual(params["idLabels"], "lab1")

    @patch("trello_helper._trello_put")
    def test_update_card_skips_none_fields(self, mock_put):
        mock_put.return_value = {"id": "c1"}
        trello_helper.update_card("c1", name="Only title")
        args, kwargs = mock_put.call_args
        params = kwargs.get("params", {})
        self.assertEqual(list(params.keys()), ["name"])

    @patch("trello_helper._trello_put")
    def test_move_card_sends_idList(self, mock_put):
        mock_put.return_value = {"id": "c1"}
        trello_helper.move_card("c1", "l2")
        args, kwargs = mock_put.call_args
        self.assertEqual(args[0], "cards/c1")
        self.assertEqual(kwargs.get("params", {}), {"idList": "l2"})

    @patch("trello_helper._trello_put")
    def test_archive_card_sets_closed_true(self, mock_put):
        mock_put.return_value = {"id": "c1"}
        trello_helper.archive_card("c1")
        args, kwargs = mock_put.call_args
        self.assertEqual(args[0], "cards/c1")
        self.assertEqual(kwargs.get("params", {}), {"closed": "true"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trello_write.TestCardWriteWrappers -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement the four wrappers**

Replace the stubs:

```python
def _format_due(due_date):
    """Convert YYYY-MM-DD to the ISO 8601 Trello expects.
    10:00 AEST (midnight UTC) keeps the due date visually on the intended day.
    """
    if not due_date:
        return None
    return f"{due_date}T10:00:00.000Z"


def create_card(board_id, list_id, title, due_date=None, description=None, label_ids=None):
    params = {"idList": list_id, "name": title}
    due = _format_due(due_date)
    if due:
        params["due"] = due
    if description:
        params["desc"] = description
    if label_ids:
        params["idLabels"] = ",".join(label_ids)
    return _trello_post("cards", params=params)


def update_card(card_id, name=None, due_date=None, description=None, label_ids=None):
    params = {}
    if name is not None:
        params["name"] = name
    if due_date is not None:
        params["due"] = _format_due(due_date)
    if description is not None:
        params["desc"] = description
    if label_ids is not None:
        params["idLabels"] = ",".join(label_ids)
    if not params:
        return None
    return _trello_put(f"cards/{card_id}", params=params)


def move_card(card_id, dest_list_id):
    return _trello_put(f"cards/{card_id}", params={"idList": dest_list_id})


def archive_card(card_id):
    return _trello_put(f"cards/{card_id}", params={"closed": "true"})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_trello_write.TestCardWriteWrappers -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the full test module to make sure nothing regressed**

Run: `python3 -m unittest tests.test_trello_write tests.test_membership_movement -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add trello_helper.py tests/test_trello_write.py
git commit -m "Implement Trello card write wrappers: create/update/move/archive"
```

---

## Task 7: Thread `raw_number` through `handle_tool_call` and add `USER_TRELLO` gate

**Files:**
- Modify: `app.py` (`handle_tool_call` signature, the call site around line 801, `_get_tools_for_user`, add `USER_TRELLO` near `USER_EMAILS`)

- [ ] **Step 1: Add `USER_TRELLO` dict**

In `app.py`, insert after the `USER_CALENDAR` block (around line 66):

```python
# Per-user Trello config. Only users in this dict get Trello write access.
USER_TRELLO = {
    "+61421188443": {  # Erin
        "board": "HIIT Office",
        "todo_list": "To Do List",
        "done_list": "Done",
    },
}
```

- [ ] **Step 2: Change `_get_tools_for_user` signature to accept the phone number**

Replace the existing function (around line 482):

```python
def _get_tools_for_user(user_email, raw_number=None):
    """Return only the tools relevant to this user — saves input tokens for non-Gmail users."""
    tools = _MINDBODY_TOOLS + _CALENDAR_TOOLS + _OUTLOOK_TOOLS
    if user_email and any(user_email == USER_EMAILS.get(phone) for phone in USER_GMAIL):
        tools = tools + _GMAIL_TOOLS
    if raw_number in USER_TRELLO:
        tools = tools + _TRELLO_WRITE_TOOLS
    return tools
```

Also update `ALL_TOOLS` (around line 492) to include the new list:

```python
ALL_TOOLS = _MINDBODY_TOOLS + _CALENDAR_TOOLS + _OUTLOOK_TOOLS + _GMAIL_TOOLS + _TRELLO_WRITE_TOOLS
```

Add a placeholder `_TRELLO_WRITE_TOOLS = []` just above `_get_tools_for_user` — we'll populate it in Task 8:

```python
_TRELLO_WRITE_TOOLS = []  # populated in a later task
```

- [ ] **Step 3: Change `handle_tool_call` signature to accept `raw_number`**

Replace the existing signature (around line 511):

```python
def handle_tool_call(tool_name, tool_input, user_email=None, raw_number=None):
```

- [ ] **Step 4: Update the call site and `tools` resolution in the webhook path**

Find the existing call around line 773 and 801 and update:

Line 773 area — change:

```python
tools = _get_tools_for_user(user_email)
```

to:

```python
tools = _get_tools_for_user(user_email, raw_number=raw_number)
```

Line 801 area — change:

```python
result = handle_tool_call(block.name, block.input, user_email=user_email)
```

to:

```python
result = handle_tool_call(block.name, block.input, user_email=user_email, raw_number=raw_number)
```

- [ ] **Step 5: Sanity-check by starting the app**

Run: `python3 -c "import app; print('ok')"`
Expected: prints `ok` (no import errors).

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "Thread raw_number through tool dispatch, add USER_TRELLO gate"
```

---

## Task 8: Add `_TRELLO_WRITE_TOOLS` definitions

**Files:**
- Modify: `app.py` (replace the `_TRELLO_WRITE_TOOLS = []` placeholder)

- [ ] **Step 1: Replace the placeholder with the 4 tool schemas**

Replace `_TRELLO_WRITE_TOOLS = []` with:

```python
_TRELLO_WRITE_TOOLS = [
    {
        "name": "add_trello_card",
        "description": (
            "Add a new card to the user's Trello board. Defaults to the user's To Do list "
            "unless list_name is specified. Use this when the user says 'add a task', "
            "'add to my to do list', 'create a card', etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Card title"},
                "due_date": {"type": "string", "description": "YYYY-MM-DD, optional"},
                "description": {"type": "string", "description": "Free text, optional"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Label names, optional. Created on the board if missing.",
                },
                "list_name": {
                    "type": "string",
                    "description": "Override list name. Defaults to the user's todo_list.",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "edit_trello_card",
        "description": (
            "Edit an existing Trello card by fuzzy title match on the user's board. "
            "Provide the current title (or a distinctive fragment) plus any fields to change. "
            "If multiple cards match, the tool returns the matches and asks the user to be more specific."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Current card title (fuzzy match)"},
                "new_title": {"type": "string", "description": "Rename the card"},
                "due_date": {"type": "string", "description": "YYYY-MM-DD"},
                "description": {"type": "string"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Replaces the card's labels with this list.",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "move_trello_card",
        "description": (
            "Move a card to another list. Defaults to the user's Done list — use this "
            "when the user says 'I've completed X', 'I've done X', 'tick off X', or 'mark X done'. "
            "Do NOT use remove_trello_card for completion — use this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Card title to find (fuzzy)"},
                "list_name": {
                    "type": "string",
                    "description": "Override destination list. Defaults to the user's done_list.",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "remove_trello_card",
        "description": (
            "Archive (delete) a Trello card. Two-step flow: call first WITHOUT confirmed=true to "
            "preview the card; then only call again with confirmed=true after the user has replied "
            "'yes' (or similar) to the preview. NEVER call with confirmed=true on the first attempt. "
            "Use only when the user explicitly says delete/archive/remove — NOT for 'completed' or 'done'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Card title to find (fuzzy)"},
                "confirmed": {"type": "boolean", "description": "Must be true on the second call after user confirms"},
            },
            "required": ["title"],
        },
    },
]
```

- [ ] **Step 2: Sanity check import**

Run: `python3 -c "import app; print(len(app._TRELLO_WRITE_TOOLS))"`
Expected: `4`.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "Add Trello write tool schemas"
```

---

## Task 9: Implement `add_trello_card` handler

**Files:**
- Modify: `app.py` (`handle_tool_call`, add a new elif branch after the existing `get_trello_tasks` branch around line 641)
- Modify: `tests/test_trello_write.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_trello_write.py`:

```python
class TestAddTrelloCardHandler(unittest.TestCase):
    def setUp(self):
        import app
        self.app = app

    @patch("trello_helper.create_card")
    @patch("trello_helper._resolve_labels")
    @patch("trello_helper._find_list")
    @patch("trello_helper._find_board_id")
    def test_adds_card_with_defaults(self, mock_board, mock_list, mock_labels, mock_create):
        mock_board.return_value = "b1"
        mock_list.return_value = ("l1", "To Do List")
        mock_labels.return_value = []
        mock_create.return_value = {"id": "c1", "shortUrl": "https://trello.com/c/c1"}

        result = self.app.handle_tool_call(
            "add_trello_card",
            {"title": "Buy kettlebells"},
            raw_number="+61421188443",
        )
        self.assertIn("Buy kettlebells", result)
        self.assertIn("To Do List", result)
        self.assertIn("HIIT Office", result)
        mock_list.assert_called_with("b1", "To Do List")
        mock_create.assert_called_once()

    @patch("trello_helper._find_board_id")
    def test_refuses_unconfigured_user(self, mock_board):
        result = self.app.handle_tool_call(
            "add_trello_card",
            {"title": "X"},
            raw_number="+61400000000",  # not in USER_TRELLO
        )
        self.assertIn("not configured", result.lower())
        mock_board.assert_not_called()

    @patch("trello_helper._find_list")
    @patch("trello_helper._find_board_id")
    def test_uses_list_name_override(self, mock_board, mock_list):
        mock_board.return_value = "b1"
        mock_list.return_value = ("l9", "Revenue Growth Ideas")
        with patch("trello_helper.create_card") as mock_create, \
             patch("trello_helper._resolve_labels", return_value=[]):
            mock_create.return_value = {"id": "c1", "shortUrl": ""}
            self.app.handle_tool_call(
                "add_trello_card",
                {"title": "X", "list_name": "Revenue Growth Ideas"},
                raw_number="+61421188443",
            )
        mock_list.assert_called_with("b1", "Revenue Growth Ideas")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trello_write.TestAddTrelloCardHandler -v`
Expected: FAIL (handler not added yet → returns the generic "unknown tool" / None result).

- [ ] **Step 3: Add the handler**

In `app.py`, after the existing `get_trello_tasks` branch (around line 641), add:

```python
    elif tool_name == "add_trello_card":
        from trello_helper import _find_board_id, _find_list, _resolve_labels, create_card
        cfg = USER_TRELLO.get(raw_number)
        if not cfg:
            return "Trello write access is not configured for you."
        board_id = _find_board_id(cfg["board"])
        if not board_id:
            return f"Could not find Trello board '{cfg['board']}'."
        list_name = tool_input.get("list_name") or cfg["todo_list"]
        list_result = _find_list(board_id, list_name)
        if not list_result:
            return f"No list matching '{list_name}' on {cfg['board']}."
        list_id, canonical_list_name = list_result
        label_ids = _resolve_labels(board_id, tool_input.get("labels") or [])
        card = create_card(
            board_id=board_id,
            list_id=list_id,
            title=tool_input["title"],
            due_date=tool_input.get("due_date"),
            description=tool_input.get("description"),
            label_ids=label_ids,
        )
        url = card.get("shortUrl", "")
        msg = f"Added '{tool_input['title']}' to {canonical_list_name} on {cfg['board']}."
        if url:
            msg += f"\n{url}"
        return msg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_trello_write.TestAddTrelloCardHandler -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_trello_write.py
git commit -m "Add add_trello_card tool handler"
```

---

## Task 10: Implement `move_trello_card` handler

**Files:**
- Modify: `app.py` (add branch after `add_trello_card`)
- Modify: `tests/test_trello_write.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_trello_write.py`:

```python
class TestMoveTrelloCardHandler(unittest.TestCase):
    def setUp(self):
        import app
        self.app = app

    @patch("trello_helper.move_card")
    @patch("trello_helper._find_cards")
    @patch("trello_helper._find_list")
    @patch("trello_helper._find_board_id")
    def test_moves_single_match_to_done(self, mock_board, mock_list, mock_find, mock_move):
        mock_board.return_value = "b1"
        mock_list.return_value = ("l_done", "Done")
        mock_find.return_value = [
            {"id": "c1", "name": "Buy kettlebells", "list_name": "To Do List",
             "list_id": "l1", "url": "https://trello.com/c/c1", "score": 15},
        ]

        result = self.app.handle_tool_call(
            "move_trello_card", {"title": "kettlebell"}, raw_number="+61421188443",
        )
        self.assertIn("Moved", result)
        self.assertIn("Buy kettlebells", result)
        self.assertIn("Done", result)
        mock_move.assert_called_with("c1", "l_done")

    @patch("trello_helper._find_cards")
    @patch("trello_helper._find_list")
    @patch("trello_helper._find_board_id")
    def test_multiple_matches_asks_to_clarify(self, mock_board, mock_list, mock_find):
        mock_board.return_value = "b1"
        mock_list.return_value = ("l_done", "Done")
        mock_find.return_value = [
            {"id": "c1", "name": "Buy kettlebells", "list_name": "To Do List",
             "list_id": "l1", "url": "", "score": 15},
            {"id": "c2", "name": "kettlebell rack install", "list_name": "Doing",
             "list_id": "l2", "url": "", "score": 15},
        ]

        result = self.app.handle_tool_call(
            "move_trello_card", {"title": "kettlebell"}, raw_number="+61421188443",
        )
        self.assertIn("Buy kettlebells", result)
        self.assertIn("kettlebell rack install", result)
        self.assertIn("more specific", result.lower())

    @patch("trello_helper._find_cards")
    @patch("trello_helper._find_list")
    @patch("trello_helper._find_board_id")
    def test_no_match_returns_friendly_message(self, mock_board, mock_list, mock_find):
        mock_board.return_value = "b1"
        mock_list.return_value = ("l_done", "Done")
        mock_find.return_value = []
        result = self.app.handle_tool_call(
            "move_trello_card", {"title": "xyz"}, raw_number="+61421188443",
        )
        self.assertIn("No card matching", result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trello_write.TestMoveTrelloCardHandler -v`
Expected: FAIL.

- [ ] **Step 3: Add the handler**

After the `add_trello_card` branch, add:

```python
    elif tool_name == "move_trello_card":
        from trello_helper import _find_board_id, _find_list, _find_cards, move_card
        cfg = USER_TRELLO.get(raw_number)
        if not cfg:
            return "Trello write access is not configured for you."
        board_id = _find_board_id(cfg["board"])
        if not board_id:
            return f"Could not find Trello board '{cfg['board']}'."
        dest_list_name = tool_input.get("list_name") or cfg["done_list"]
        dest = _find_list(board_id, dest_list_name)
        if not dest:
            return f"No list matching '{dest_list_name}' on {cfg['board']}."
        dest_id, dest_canonical = dest
        matches = _find_cards(board_id, tool_input.get("title", ""))
        if not matches:
            return f"No card matching '{tool_input.get('title')}' on {cfg['board']}."
        if len(matches) > 1:
            top = matches[:5]
            lines = [f"Multiple matches — please be more specific:"]
            for m in top:
                lines.append(f"  - {m['name']} ({m['list_name']})")
            return "\n".join(lines)
        m = matches[0]
        move_card(m["id"], dest_id)
        return f"Moved '{m['name']}' from {m['list_name']} to {dest_canonical}."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_trello_write.TestMoveTrelloCardHandler -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_trello_write.py
git commit -m "Add move_trello_card tool handler"
```

---

## Task 11: Implement `edit_trello_card` handler

**Files:**
- Modify: `app.py` (add branch after `move_trello_card`)
- Modify: `tests/test_trello_write.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_trello_write.py`:

```python
class TestEditTrelloCardHandler(unittest.TestCase):
    def setUp(self):
        import app
        self.app = app

    @patch("trello_helper.update_card")
    @patch("trello_helper._resolve_labels")
    @patch("trello_helper._find_cards")
    @patch("trello_helper._find_board_id")
    def test_updates_single_match(self, mock_board, mock_find, mock_labels, mock_update):
        mock_board.return_value = "b1"
        mock_find.return_value = [
            {"id": "c1", "name": "Buy kettlebells", "list_name": "To Do List",
             "list_id": "l1", "url": "", "score": 15},
        ]
        mock_labels.return_value = []

        result = self.app.handle_tool_call(
            "edit_trello_card",
            {"title": "kettlebell", "new_title": "Buy heavier kettlebells",
             "due_date": "2026-05-01"},
            raw_number="+61421188443",
        )
        self.assertIn("Updated", result)
        args, kwargs = mock_update.call_args
        self.assertEqual(args[0], "c1")
        self.assertEqual(kwargs["name"], "Buy heavier kettlebells")
        self.assertEqual(kwargs["due_date"], "2026-05-01")

    @patch("trello_helper._find_cards")
    @patch("trello_helper._find_board_id")
    def test_multiple_matches_asks_to_clarify(self, mock_board, mock_find):
        mock_board.return_value = "b1"
        mock_find.return_value = [
            {"id": "c1", "name": "Buy kettlebells", "list_name": "To Do List",
             "list_id": "l1", "url": "", "score": 15},
            {"id": "c2", "name": "kettlebell rack install", "list_name": "Doing",
             "list_id": "l2", "url": "", "score": 15},
        ]
        result = self.app.handle_tool_call(
            "edit_trello_card",
            {"title": "kettlebell", "new_title": "renamed"},
            raw_number="+61421188443",
        )
        self.assertIn("more specific", result.lower())

    @patch("trello_helper._find_cards")
    @patch("trello_helper._find_board_id")
    def test_no_match(self, mock_board, mock_find):
        mock_board.return_value = "b1"
        mock_find.return_value = []
        result = self.app.handle_tool_call(
            "edit_trello_card",
            {"title": "abc", "new_title": "def"},
            raw_number="+61421188443",
        )
        self.assertIn("No card matching", result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trello_write.TestEditTrelloCardHandler -v`
Expected: FAIL.

- [ ] **Step 3: Add the handler**

After the `move_trello_card` branch, add:

```python
    elif tool_name == "edit_trello_card":
        from trello_helper import _find_board_id, _find_cards, _resolve_labels, update_card
        cfg = USER_TRELLO.get(raw_number)
        if not cfg:
            return "Trello write access is not configured for you."
        board_id = _find_board_id(cfg["board"])
        if not board_id:
            return f"Could not find Trello board '{cfg['board']}'."
        matches = _find_cards(board_id, tool_input.get("title", ""))
        if not matches:
            return f"No card matching '{tool_input.get('title')}' on {cfg['board']}."
        if len(matches) > 1:
            top = matches[:5]
            lines = ["Multiple matches — please be more specific:"]
            for m in top:
                lines.append(f"  - {m['name']} ({m['list_name']})")
            return "\n".join(lines)
        m = matches[0]

        # Resolve labels if given (empty list means "clear labels"; None means "don't touch")
        label_names = tool_input.get("labels")
        label_ids = None
        if label_names is not None:
            label_ids = _resolve_labels(board_id, label_names)

        update_card(
            m["id"],
            name=tool_input.get("new_title"),
            due_date=tool_input.get("due_date"),
            description=tool_input.get("description"),
            label_ids=label_ids,
        )
        return f"Updated '{m['name']}'."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_trello_write.TestEditTrelloCardHandler -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_trello_write.py
git commit -m "Add edit_trello_card tool handler"
```

---

## Task 12: Implement `remove_trello_card` handler (two-step flow)

**Files:**
- Modify: `app.py` (add branch after `edit_trello_card`)
- Modify: `tests/test_trello_write.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_trello_write.py`:

```python
class TestRemoveTrelloCardHandler(unittest.TestCase):
    def setUp(self):
        import app
        self.app = app

    @patch("trello_helper.archive_card")
    @patch("trello_helper._find_cards")
    @patch("trello_helper._find_board_id")
    def test_preview_does_not_archive(self, mock_board, mock_find, mock_archive):
        mock_board.return_value = "b1"
        mock_find.return_value = [
            {"id": "c1", "name": "Buy kettlebells", "list_name": "To Do List",
             "list_id": "l1", "url": "", "score": 15},
        ]
        result = self.app.handle_tool_call(
            "remove_trello_card", {"title": "kettlebell"}, raw_number="+61421188443",
        )
        self.assertIn("Buy kettlebells", result)
        self.assertIn("yes", result.lower())
        mock_archive.assert_not_called()

    @patch("trello_helper.archive_card")
    @patch("trello_helper._find_cards")
    @patch("trello_helper._find_board_id")
    def test_confirmed_archives(self, mock_board, mock_find, mock_archive):
        mock_board.return_value = "b1"
        mock_find.return_value = [
            {"id": "c1", "name": "Buy kettlebells", "list_name": "To Do List",
             "list_id": "l1", "url": "", "score": 15},
        ]
        result = self.app.handle_tool_call(
            "remove_trello_card",
            {"title": "kettlebell", "confirmed": True},
            raw_number="+61421188443",
        )
        self.assertIn("Archived", result)
        mock_archive.assert_called_with("c1")

    @patch("trello_helper._find_cards")
    @patch("trello_helper._find_board_id")
    def test_multiple_matches_asks_to_clarify_even_with_confirmed(self, mock_board, mock_find):
        mock_board.return_value = "b1"
        mock_find.return_value = [
            {"id": "c1", "name": "Buy kettlebells", "list_name": "To Do List",
             "list_id": "l1", "url": "", "score": 15},
            {"id": "c2", "name": "kettlebell rack install", "list_name": "Doing",
             "list_id": "l2", "url": "", "score": 15},
        ]
        result = self.app.handle_tool_call(
            "remove_trello_card",
            {"title": "kettlebell", "confirmed": True},
            raw_number="+61421188443",
        )
        self.assertIn("more specific", result.lower())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_trello_write.TestRemoveTrelloCardHandler -v`
Expected: FAIL.

- [ ] **Step 3: Add the handler**

After the `edit_trello_card` branch, add:

```python
    elif tool_name == "remove_trello_card":
        from trello_helper import _find_board_id, _find_cards, archive_card
        cfg = USER_TRELLO.get(raw_number)
        if not cfg:
            return "Trello write access is not configured for you."
        board_id = _find_board_id(cfg["board"])
        if not board_id:
            return f"Could not find Trello board '{cfg['board']}'."
        matches = _find_cards(board_id, tool_input.get("title", ""))
        if not matches:
            return f"No card matching '{tool_input.get('title')}' on {cfg['board']}."
        if len(matches) > 1:
            top = matches[:5]
            lines = ["Multiple matches — please be more specific:"]
            for m in top:
                lines.append(f"  - {m['name']} ({m['list_name']})")
            return "\n".join(lines)
        m = matches[0]
        if not tool_input.get("confirmed"):
            return (
                f"Found '{m['name']}' in {m['list_name']}. "
                f"Reply 'yes' to archive."
            )
        archive_card(m["id"])
        return f"Archived '{m['name']}'."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_trello_write.TestRemoveTrelloCardHandler -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Full test run**

Run: `python3 -m unittest discover tests -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_trello_write.py
git commit -m "Add remove_trello_card tool with two-step confirm flow"
```

---

## Task 13: Update `SYSTEM_PROMPT`

**Files:**
- Modify: `app.py` (`SYSTEM_PROMPT` around lines 68-109)

- [ ] **Step 1: Add Trello write guidance to the system prompt**

In `app.py`, find the sentence in `SYSTEM_PROMPT` that currently says:

```
"Trello (HIIT Challenge board tasks), "
```

Replace that fragment with:

```
"Trello (read HIIT Challenge board cards; and — for users with write access — add, edit, move, and archive cards on their configured board), "
```

Then, just before the closing `)` of `SYSTEM_PROMPT`, append these additional instruction sentences:

```python
    "IMPORTANT — Trello write tools: When the user says 'I've completed X', 'I've done X', "
    "'tick off X', or 'mark X done', use move_trello_card (it defaults to the user's Done list). "
    "Do NOT use remove_trello_card for completion — that is only for explicit delete/archive/remove. "
    "For remove_trello_card, you MUST follow a two-step flow: first call WITHOUT confirmed=true to "
    "show the user what will be archived, wait for an explicit 'yes' or 'confirm' in the next user "
    "message, then call again with confirmed=true. NEVER pass confirmed=true on the first call. "
    "For add_trello_card, if the user doesn't name a list, leave list_name unset — the tool defaults "
    "to their To Do list. "
```

- [ ] **Step 2: Sanity check import**

Run: `python3 -c "import app; print('Trello' in app.SYSTEM_PROMPT and 'confirmed=true' in app.SYSTEM_PROMPT)"`
Expected: `True`.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "Update SYSTEM_PROMPT with Trello write guidance"
```

---

## Task 14: Deploy and smoke-test

**Files:** (no changes — deploy only)

- [ ] **Step 1: Run the full test suite one more time**

Run: `python3 -m unittest discover tests -v`
Expected: All PASS, no errors.

- [ ] **Step 2: Deploy**

Run: `railway up --detach`
Expected: Upload succeeds, build logs URL printed.

- [ ] **Step 3: Wait for deploy and tail logs for startup**

Run: `railway logs --deployment 2>&1 | tail -20`
Expected: See `Listening at: http://0.0.0.0:5000` and `App loaded.` with no tracebacks.

- [ ] **Step 4: Ask the user to smoke-test each operation via WhatsApp from Erin's phone**

Tell the user these are the things to try (from Erin's WhatsApp):

1. **Add** — send: *"add a card 'test task from bot' to my to do list due Friday"*
   Expect: card appears in "To Do List" on HIIT Office, bot replies with a Trello link.
2. **Edit** — send: *"edit 'test task' — change title to 'test task updated'"*
   Expect: bot replies *"Updated 'test task from bot'."* Card title is now *"test task updated"*.
3. **Move** — send: *"I've completed the test task"*
   Expect: card moves to "Done" list, bot replies *"Moved 'test task updated' from To Do List to Done."*
4. **Remove (step 1)** — send: *"delete the test task"*
   Expect: bot replies *"Found 'test task updated' in Done. Reply 'yes' to archive."* (card still there)
5. **Remove (step 2)** — reply *"yes"*
   Expect: bot replies *"Archived 'test task updated'."* Card is archived in Trello.

- [ ] **Step 5: After user confirms all 5 steps work, final commit note (if anything was fixed during smoke test)**

If no fixes needed: no commit. If anything needed fixing, commit as normal.

---

## Done

All four Trello write operations implemented, tested, deployed. Erin can manage her HIIT Office board cards from WhatsApp.
