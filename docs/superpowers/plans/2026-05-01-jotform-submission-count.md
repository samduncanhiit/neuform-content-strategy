# JotForm Submission Count Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a WhatsApp bot tool `get_jotform_submissions` that, given a form name, returns the total submission count for the matched JotForm form. Available to all users.

**Architecture:** New `jotform_helper.py` (mirrors `trello_helper.py` shape) wraps the JotForm REST API. A single `/user/forms` call returns each form's `count` field, so no second call is needed. `app.py` adds the tool def, wires it into `handle_tool_call`, adds it to the unconditional tool bundle in `_get_tools_for_user`, and updates `SYSTEM_PROMPT`.

**Tech Stack:** Python 3, `requests` (already in `requirements.txt`), `unittest` + `unittest.mock` (matches existing `tests/test_trello_write.py` style).

**Spec:** `docs/superpowers/specs/2026-05-01-jotform-submission-count-design.md`

---

## File Structure

| Path | Action | Responsibility |
|------|--------|----------------|
| `jotform_helper.py` | Create | JotForm REST wrapper + `get_submission_count` |
| `tests/test_jotform_helper.py` | Create | Unit tests for the helper |
| `app.py` | Modify | Add tool def, handler branch, user-tool wiring, SYSTEM_PROMPT update |

`requirements.txt` is unchanged — `requests==2.32.3` is already present.

---

## Return Shape (used across tasks)

`get_submission_count(name)` always returns a dict with a `status` field:

```python
{"status": "ok", "title": str, "count": int}
{"status": "none", "name": str}
{"status": "multiple", "matches": [{"title": str, "count": int}, ...]}
{"status": "error", "message": str}
```

The `app.py` handler maps each to a WhatsApp-friendly string.

---

## Task 1: Scaffold helper module + scaffolding test

**Files:**
- Create: `jotform_helper.py`
- Create: `tests/test_jotform_helper.py`

- [ ] **Step 1: Write the failing scaffolding test**

Create `tests/test_jotform_helper.py`:

```python
"""Unit tests for jotform_helper."""
import unittest
from unittest.mock import patch

import jotform_helper


class TestScaffolding(unittest.TestCase):
    def test_module_exposes_public_helpers(self):
        for name in [
            "_api_get",
            "_find_forms_by_name",
            "get_submission_count",
        ]:
            self.assertTrue(
                hasattr(jotform_helper, name),
                f"missing {name}",
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_jotform_helper -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jotform_helper'`

- [ ] **Step 3: Create the minimal helper module**

Create `jotform_helper.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_jotform_helper -v`
Expected: PASS — scaffolding test green.

- [ ] **Step 5: Commit**

```bash
git add jotform_helper.py tests/test_jotform_helper.py
git commit -m "feat: scaffold jotform_helper module"
```

---

## Task 2: Implement `_api_get`

**Files:**
- Modify: `jotform_helper.py`
- Modify: `tests/test_jotform_helper.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_jotform_helper.py`:

```python
class TestApiGet(unittest.TestCase):
    @patch("jotform_helper.JOTFORM_API_KEY", "fake-key")
    @patch("jotform_helper.requests.get")
    def test_calls_correct_url_with_api_key(self, mock_get):
        mock_resp = mock_get.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"content": []}
        jotform_helper._api_get("/user/forms", {"limit": 1000})
        args, kwargs = mock_get.call_args
        self.assertEqual(args[0], "https://api.jotform.com/user/forms")
        self.assertEqual(kwargs["params"]["apiKey"], "fake-key")
        self.assertEqual(kwargs["params"]["limit"], 1000)

    @patch("jotform_helper.JOTFORM_API_KEY", None)
    def test_raises_when_api_key_missing(self):
        with self.assertRaises(RuntimeError) as cm:
            jotform_helper._api_get("/user/forms")
        self.assertIn("JOTFORM_API_KEY", str(cm.exception))

    @patch("jotform_helper.JOTFORM_API_KEY", "fake-key")
    @patch("jotform_helper.requests.get")
    def test_returns_content_field_from_response(self, mock_get):
        mock_resp = mock_get.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "responseCode": 200,
            "content": [{"id": "f1", "title": "A"}],
        }
        result = jotform_helper._api_get("/user/forms")
        self.assertEqual(result, [{"id": "f1", "title": "A"}])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_jotform_helper.TestApiGet -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `_api_get`**

Replace the `_api_get` stub in `jotform_helper.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_jotform_helper.TestApiGet -v`
Expected: PASS — 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add jotform_helper.py tests/test_jotform_helper.py
git commit -m "feat: implement jotform _api_get with API key handling"
```

---

## Task 3: Implement `_find_forms_by_name`

**Files:**
- Modify: `jotform_helper.py`
- Modify: `tests/test_jotform_helper.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_jotform_helper.py`:

```python
def _form(form_id, title, count, status="ENABLED"):
    return {"id": form_id, "title": title, "count": str(count), "status": status}


class TestFindFormsByName(unittest.TestCase):
    @patch("jotform_helper._api_get")
    def test_exact_match_case_insensitive(self, mock_get):
        mock_get.return_value = [
            _form("1", "Lead Form", 12),
            _form("2", "Other Form", 5),
        ]
        result = jotform_helper._find_forms_by_name("lead form")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "1")

    @patch("jotform_helper._api_get")
    def test_substring_match_when_no_exact(self, mock_get):
        mock_get.return_value = [
            _form("1", "Lead Form V1", 12),
            _form("2", "Lead Form V2", 7),
            _form("3", "Other Form", 5),
        ]
        result = jotform_helper._find_forms_by_name("lead form")
        self.assertEqual({f["id"] for f in result}, {"1", "2"})

    @patch("jotform_helper._api_get")
    def test_no_match_returns_empty(self, mock_get):
        mock_get.return_value = [_form("1", "Other Form", 5)]
        self.assertEqual(jotform_helper._find_forms_by_name("nope"), [])

    @patch("jotform_helper._api_get")
    def test_filters_out_deleted_forms(self, mock_get):
        mock_get.return_value = [
            _form("1", "Lead Form", 12, status="DELETED"),
            _form("2", "Lead Form", 7, status="ENABLED"),
        ]
        result = jotform_helper._find_forms_by_name("lead form")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "2")

    @patch("jotform_helper._api_get")
    def test_exact_match_takes_priority_over_substring(self, mock_get):
        mock_get.return_value = [
            _form("1", "Lead Form", 12),
            _form("2", "Lead Form V2", 7),
        ]
        result = jotform_helper._find_forms_by_name("lead form")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "1")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_jotform_helper.TestFindFormsByName -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `_find_forms_by_name`**

Replace the stub in `jotform_helper.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_jotform_helper.TestFindFormsByName -v`
Expected: PASS — 5 tests green.

- [ ] **Step 5: Commit**

```bash
git add jotform_helper.py tests/test_jotform_helper.py
git commit -m "feat: implement jotform form lookup by name"
```

---

## Task 4: Implement `get_submission_count`

**Files:**
- Modify: `jotform_helper.py`
- Modify: `tests/test_jotform_helper.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_jotform_helper.py`:

```python
class TestGetSubmissionCount(unittest.TestCase):
    @patch("jotform_helper._find_forms_by_name")
    def test_single_match_returns_ok(self, mock_find):
        mock_find.return_value = [_form("1", "Lead Form", 42)]
        result = jotform_helper.get_submission_count("lead form")
        self.assertEqual(result, {
            "status": "ok",
            "title": "Lead Form",
            "count": 42,
        })

    @patch("jotform_helper._find_forms_by_name")
    def test_no_match_returns_none_status(self, mock_find):
        mock_find.return_value = []
        result = jotform_helper.get_submission_count("nope")
        self.assertEqual(result, {"status": "none", "name": "nope"})

    @patch("jotform_helper._find_forms_by_name")
    def test_multiple_matches_returns_multiple(self, mock_find):
        mock_find.return_value = [
            _form("1", "Lead Form V1", 12),
            _form("2", "Lead Form V2", 7),
        ]
        result = jotform_helper.get_submission_count("lead form")
        self.assertEqual(result["status"], "multiple")
        self.assertEqual(result["matches"], [
            {"title": "Lead Form V1", "count": 12},
            {"title": "Lead Form V2", "count": 7},
        ])

    @patch("jotform_helper._find_forms_by_name")
    def test_missing_api_key_returns_error(self, mock_find):
        mock_find.side_effect = RuntimeError("JOTFORM_API_KEY env var is not set")
        result = jotform_helper.get_submission_count("anything")
        self.assertEqual(result["status"], "error")
        self.assertIn("not configured", result["message"].lower())

    @patch("jotform_helper._find_forms_by_name")
    def test_http_401_returns_error(self, mock_find):
        from requests import HTTPError, Response
        resp = Response()
        resp.status_code = 401
        mock_find.side_effect = HTTPError(response=resp)
        result = jotform_helper.get_submission_count("anything")
        self.assertEqual(result["status"], "error")
        self.assertIn("invalid", result["message"].lower())

    @patch("jotform_helper._find_forms_by_name")
    def test_network_error_returns_error(self, mock_find):
        from requests import ConnectionError as ReqConnErr
        mock_find.side_effect = ReqConnErr("boom")
        result = jotform_helper.get_submission_count("anything")
        self.assertEqual(result["status"], "error")
        self.assertIn("try again", result["message"].lower())

    @patch("jotform_helper._find_forms_by_name")
    def test_count_is_coerced_to_int(self, mock_find):
        # JotForm returns count as a string — make sure we coerce.
        mock_find.return_value = [_form("1", "Lead Form", "99")]
        result = jotform_helper.get_submission_count("lead form")
        self.assertEqual(result["count"], 99)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_jotform_helper.TestGetSubmissionCount -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `get_submission_count`**

Replace the stub in `jotform_helper.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_jotform_helper -v`
Expected: PASS — all helper tests green (scaffolding + api_get + find + get_submission_count = 16 tests).

- [ ] **Step 5: Commit**

```bash
git add jotform_helper.py tests/test_jotform_helper.py
git commit -m "feat: implement get_submission_count with error handling"
```

---

## Task 5: Add tool definition and handler in `app.py`

**Files:**
- Modify: `app.py` (tool list + `handle_tool_call` + tool group bundle + `ALL_TOOLS`)

- [ ] **Step 1: Add the tool group list**

In `app.py`, locate `_TRELLO_WRITE_TOOLS = [` (around line 499). Immediately AFTER the closing `]` of `_TRELLO_WRITE_TOOLS`, add:

```python
_JOTFORM_TOOLS = [
    {
        "name": "get_jotform_submissions",
        "description": (
            "Get the total submission count for a JotForm form, looked up by form name. "
            "Use this when the user asks 'how many submissions for X', 'submission count', "
            "or similar. Matches form name case-insensitively. If multiple forms match, "
            "the bot will list them so the user can pick."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "form_name": {
                    "type": "string",
                    "description": "Name (or partial name) of the JotForm form.",
                },
            },
            "required": ["form_name"],
        },
    },
]
```

- [ ] **Step 2: Wire into `_get_tools_for_user` and `ALL_TOOLS`**

Find the function `_get_tools_for_user` (around line 588). Change:

```python
def _get_tools_for_user(user_email, raw_number=None):
    """Return only the tools relevant to this user — saves ~500 input tokens for non-Gmail users."""
    tools = _MINDBODY_TOOLS + _CALENDAR_TOOLS + _OUTLOOK_TOOLS
```

to:

```python
def _get_tools_for_user(user_email, raw_number=None):
    """Return only the tools relevant to this user — saves ~500 input tokens for non-Gmail users."""
    tools = _MINDBODY_TOOLS + _CALENDAR_TOOLS + _OUTLOOK_TOOLS + _JOTFORM_TOOLS
```

Then find the line:

```python
ALL_TOOLS = _MINDBODY_TOOLS + _CALENDAR_TOOLS + _OUTLOOK_TOOLS + _GMAIL_TOOLS + _TRELLO_WRITE_TOOLS
```

and change to:

```python
ALL_TOOLS = _MINDBODY_TOOLS + _CALENDAR_TOOLS + _OUTLOOK_TOOLS + _GMAIL_TOOLS + _TRELLO_WRITE_TOOLS + _JOTFORM_TOOLS
```

- [ ] **Step 3: Add the handler branch**

In `handle_tool_call`, find the last `elif tool_name == "remove_trello_card":` block (the Trello block ends around line 838+). Immediately AFTER that block (before the next `elif` for any other tool, or before the final `else`/`return` of the function), add:

```python
    elif tool_name == "get_jotform_submissions":
        from jotform_helper import get_submission_count
        result = get_submission_count(tool_input.get("form_name", ""))
        if result["status"] == "ok":
            return f"*{result['title']}*: {result['count']} submissions"
        elif result["status"] == "none":
            return f"No JotForm form found matching '{result['name']}'."
        elif result["status"] == "multiple":
            lines = ["Multiple forms match — which one?"]
            for m in result["matches"]:
                lines.append(f"• {m['title']} ({m['count']} submissions)")
            return "\n".join(lines)
        else:  # error
            return result["message"]
```

(If you cannot locate the right insertion point because the surrounding structure has changed, search for `elif tool_name == "get_trello_tasks":` and add the new branch immediately after the entire Trello block — order does not matter functionally.)

- [ ] **Step 4: Verify the file still parses**

Run: `python -c "import ast; ast.parse(open('app.py').read())"`
Expected: no output (clean parse).

- [ ] **Step 5: Verify tool wiring with a quick smoke check**

Run:

```bash
python -c "
import app
assert any(t['name'] == 'get_jotform_submissions' for t in app.ALL_TOOLS), 'tool not in ALL_TOOLS'
tools = app._get_tools_for_user('sam@hiitaustralia.com.au', '+61420233508')
assert any(t['name'] == 'get_jotform_submissions' for t in tools), 'tool not in user tools'
print('ok')
"
```
Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: add get_jotform_submissions tool to bot"
```

---

## Task 6: Update `SYSTEM_PROMPT`

**Files:**
- Modify: `app.py` (the `SYSTEM_PROMPT` string, around lines 95-126)

- [ ] **Step 1: Add a JotForm hint to the system prompt**

In `app.py`, find this line inside `SYSTEM_PROMPT`:

```python
    "FORMATTING: All responses must be plain text suitable for copy-pasting into other chats. "
```

Immediately BEFORE that line, add:

```python
    "IMPORTANT — JotForm: When the user asks 'how many submissions for X', 'submission count for X', "
    "'how many people filled out X', or any similar question about a JotForm form's submission count, "
    "use get_jotform_submissions with the form name they mentioned. If the bot returns multiple matches, "
    "show the list to the user verbatim and ask them to pick one. "
```

- [ ] **Step 2: Verify the file still parses**

Run: `python -c "import ast; ast.parse(open('app.py').read())"`
Expected: no output.

- [ ] **Step 3: Verify the prompt contains the hint**

Run: `python -c "import app; assert 'get_jotform_submissions' in app.SYSTEM_PROMPT; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add JotForm hint to SYSTEM_PROMPT"
```

---

## Task 7: Local end-to-end smoke test (optional but recommended)

**Files:** none modified.

This task verifies the helper hits the real JotForm API against your account. Skip if you'd rather smoke-test on Railway.

- [ ] **Step 1: Confirm you have a JotForm API key**

Get your API key from JotForm: Account → API → Create New Key.

- [ ] **Step 2: Run the helper against the live API**

Run (replace `YOUR_KEY` and pick a form name you know exists):

```bash
JOTFORM_API_KEY=YOUR_KEY python -c "
from jotform_helper import get_submission_count
import json
print(json.dumps(get_submission_count('Lower Body Strength'), indent=2))
"
```

Expected: a dict with `status: ok` (or `multiple` if your form name is ambiguous), `title`, and `count`.

- [ ] **Step 3: Try a name that doesn't exist**

```bash
JOTFORM_API_KEY=YOUR_KEY python -c "
from jotform_helper import get_submission_count
print(get_submission_count('definitely-not-a-real-form'))
"
```

Expected: `{'status': 'none', 'name': 'definitely-not-a-real-form'}`.

- [ ] **Step 4: Try with a missing key**

```bash
unset JOTFORM_API_KEY && python -c "
from jotform_helper import get_submission_count
print(get_submission_count('anything'))
"
```

Expected: `{'status': 'error', 'message': 'JotForm is not configured.'}`.

No commit — this is verification only.

---

## Task 8: Deploy

**Files:** none modified — Railway env + deploy.

- [ ] **Step 1: Add `JOTFORM_API_KEY` to Railway**

In the Railway dashboard for project `adequate-playfulness`, service `hiit-automations`, add an environment variable:

- Name: `JOTFORM_API_KEY`
- Value: your JotForm API key

Save. (The service will redeploy automatically; that's fine.)

- [ ] **Step 2: Deploy from CLI**

Run from the repo root: `railway up --detach`
Expected: deploy starts, returns immediately. Wait ~30s for it to come up.

- [ ] **Step 3: WhatsApp smoke test**

Send a WhatsApp message to the bot from any of the three users:

> "How many submissions for [form name]?"

Expected reply: `*[Form Name]*: N submissions` (or a list of matches if ambiguous, or "No JotForm form found..." if no match).

- [ ] **Step 4: Watch logs if anything looks off**

Run: `railway logs` — confirm no tracebacks tied to the new tool. If `JOTFORM_API_KEY` was forgotten you'll see `"JotForm is not configured."` in the WhatsApp reply.

No commit — deployment only.

---

## Done

All spec requirements implemented:

- ✅ New `jotform_helper.py` mirrors `trello_helper.py` shape
- ✅ Lookup by form name (case-insensitive, exact-priority then substring)
- ✅ Returns total count from `/user/forms` `count` field (single API call)
- ✅ Handles ok / none / multiple / error cases
- ✅ DELETED forms filtered out
- ✅ Tool wired to all three users (added to unconditional bundle)
- ✅ `SYSTEM_PROMPT` updated so the model knows when to call it
- ✅ Env var `JOTFORM_API_KEY` documented and required at deploy time
