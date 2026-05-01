# JotForm Closest-Match Suggestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When `get_jotform_submissions` finds no substring match for the user's query, suggest the single closest form (with its submission count) using fuzzy token-overlap scoring. Fall back to the existing "no match" message if even the best candidate scores too low.

**Architecture:** Add `_fuzzy_score` (copied verbatim from `trello_helper.py`) and `_suggest_form` to `jotform_helper.py`. `get_submission_count` calls `_suggest_form` on the no-match branch and returns a new `{"status": "suggest", ...}` shape. The `app.py` handler grows one new branch to format the suggest reply.

**Tech Stack:** Python 3, `requests`, `unittest` + `unittest.mock` (no new dependencies).

**Spec:** `docs/superpowers/specs/2026-05-01-jotform-closest-match-suggestion-design.md`

---

## File Structure

| Path | Action | Responsibility |
|------|--------|----------------|
| `jotform_helper.py` | Modify | Add `_fuzzy_score`, `_suggest_form`, `MIN_SUGGESTION_SCORE` constant; update `get_submission_count` |
| `tests/test_jotform_helper.py` | Modify | Add `TestFuzzyScore`, `TestSuggestForm`, and 2 new tests in `TestGetSubmissionCount` |
| `app.py` | Modify | Add `suggest` branch to `get_jotform_submissions` handler |

No new files. No `requirements.txt` change.

---

## Return Shape (used across tasks)

`get_submission_count(name)` gains a fifth shape:

```python
{"status": "suggest", "name": str, "suggestion": {"title": str, "count": int}}
```

The `app.py` handler maps it to:
```
No exact match for '<name>'. Did you mean *<title>*? (<count> submissions)
```

---

## Task 1: Add `_fuzzy_score` to `jotform_helper.py`

**Files:**
- Modify: `jotform_helper.py`
- Modify: `tests/test_jotform_helper.py`

- [ ] **Step 1: Append failing tests to `tests/test_jotform_helper.py`**

Append this to the END of the file (after the existing `TestGetSubmissionCount` class):

```python
class TestFuzzyScore(unittest.TestCase):
    def test_exact_match_is_highest(self):
        self.assertGreater(
            jotform_helper._fuzzy_score("lower body", "lower body"),
            jotform_helper._fuzzy_score("lower body", "upper body"),
        )

    def test_substring_match_beats_token_only(self):
        # substring scores 10+, token-only scores 1-9
        self.assertGreaterEqual(
            jotform_helper._fuzzy_score("strength", "lower body strength"),
            10,
        )

    def test_case_insensitive(self):
        self.assertEqual(
            jotform_helper._fuzzy_score("Strength", "lower body strength"),
            jotform_helper._fuzzy_score("strength", "LOWER BODY STRENGTH"),
        )

    def test_token_overlap_scores(self):
        # "challenge round" should match "8 Week Challenge Round 14" via tokens
        score = jotform_helper._fuzzy_score(
            "challenge round", "8 Week Challenge Round 14"
        )
        self.assertGreater(score, 0)

    def test_empty_needle_returns_zero(self):
        self.assertEqual(jotform_helper._fuzzy_score("", "anything"), 0)

    def test_empty_haystack_returns_zero(self):
        self.assertEqual(jotform_helper._fuzzy_score("anything", ""), 0)

    def test_no_overlap_returns_zero(self):
        self.assertEqual(
            jotform_helper._fuzzy_score("kettlebell", "labour day breakfast"),
            0,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_jotform_helper.TestFuzzyScore -v`
Expected: ERROR — `AttributeError: module 'jotform_helper' has no attribute '_fuzzy_score'` (or 7 errors).

- [ ] **Step 3: Add `_fuzzy_score` to `jotform_helper.py`**

In `jotform_helper.py`, immediately AFTER the `_find_forms_by_name` function (and before `get_submission_count`), add:

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
```

This is a verbatim copy from `trello_helper.py:144` — same algorithm, deliberately duplicated rather than extracted to keep the modules independent.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_jotform_helper.TestFuzzyScore -v`
Expected: PASS — 7 tests green.

Then full module: `python -m unittest tests.test_jotform_helper -v`
Expected: 24 tests total (17 existing + 7 new), all PASS.

- [ ] **Step 5: Commit**

```bash
git add jotform_helper.py tests/test_jotform_helper.py
git commit -m "feat: add _fuzzy_score helper to jotform_helper"
```

---

## Task 2: Add `_suggest_form` and the threshold constant

**Files:**
- Modify: `jotform_helper.py`
- Modify: `tests/test_jotform_helper.py`

- [ ] **Step 1: Append failing tests to `tests/test_jotform_helper.py`**

Append this to the END of the file (after the new `TestFuzzyScore` class):

```python
class TestSuggestForm(unittest.TestCase):
    @patch("jotform_helper._api_get")
    def test_returns_highest_scoring_form(self, mock_get):
        mock_get.return_value = [
            _form("1", "Lower Body Strength Session", 40),
            _form("2", "Upper Body Hypertrophy", 12),
            _form("3", "Cardio Bootcamp", 5),
        ]
        # "lower body strength" is a substring of form 1's title — score 10+
        result = jotform_helper._suggest_form("lower body strength")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "1")

    @patch("jotform_helper._api_get")
    def test_returns_token_match_when_no_substring(self, mock_get):
        mock_get.return_value = [
            _form("1", "8 Week Challenge Round 14", 36),
            _form("2", "Cardio Bootcamp", 5),
        ]
        # "challenge week" isn't a substring of either title (form 1 has "Week Challenge"
        # in that order), but tokens "challenge" and "week" both match form 1 → token-only
        # score = 2. Form 2 shares no tokens → score = 0. Form 1 wins.
        result = jotform_helper._suggest_form("challenge week")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "1")

    @patch("jotform_helper._api_get")
    def test_returns_none_when_no_form_scores_above_threshold(self, mock_get):
        mock_get.return_value = [
            _form("1", "Cardio Bootcamp", 5),
            _form("2", "End of Challenge Party", 0),
        ]
        # "kettlebell" shares no tokens with either title → all score 0
        result = jotform_helper._suggest_form("kettlebell")
        self.assertIsNone(result)

    @patch("jotform_helper._api_get")
    def test_filters_out_deleted_forms(self, mock_get):
        mock_get.return_value = [
            _form("1", "Lower Body Strength", 40, status="DELETED"),
            _form("2", "Cardio Bootcamp", 5),
        ]
        # The deleted form would be the closest match for "lower body" but should be skipped.
        result = jotform_helper._suggest_form("lower body")
        # Cardio Bootcamp scores 0 against "lower body", so we expect None.
        self.assertIsNone(result)

    @patch("jotform_helper._api_get")
    def test_empty_name_returns_none(self, mock_get):
        # No API call needed for empty input.
        result = jotform_helper._suggest_form("")
        self.assertIsNone(result)
        mock_get.assert_not_called()

    @patch("jotform_helper._api_get")
    def test_whitespace_only_name_returns_none(self, mock_get):
        result = jotform_helper._suggest_form("   ")
        self.assertIsNone(result)
        mock_get.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_jotform_helper.TestSuggestForm -v`
Expected: ERROR — `AttributeError: module 'jotform_helper' has no attribute '_suggest_form'` (or 6 errors).

- [ ] **Step 3: Add `MIN_SUGGESTION_SCORE` constant and `_suggest_form` function**

In `jotform_helper.py`, find the existing constant block at the top:

```python
JOTFORM_BASE = "https://api.jotform.com"
JOTFORM_API_KEY = os.environ.get("JOTFORM_API_KEY")
```

Immediately AFTER those two lines, add:

```python
MIN_SUGGESTION_SCORE = 1
```

Then, immediately AFTER the `_fuzzy_score` function (added in Task 1) and BEFORE `get_submission_count`, add:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_jotform_helper.TestSuggestForm -v`
Expected: PASS — 6 tests green.

Then full module: `python -m unittest tests.test_jotform_helper -v`
Expected: 30 tests total (24 + 6), all PASS.

- [ ] **Step 5: Commit**

```bash
git add jotform_helper.py tests/test_jotform_helper.py
git commit -m "feat: add _suggest_form for fuzzy fallback lookup"
```

---

## Task 3: Wire `_suggest_form` into `get_submission_count`

**Files:**
- Modify: `jotform_helper.py`
- Modify: `tests/test_jotform_helper.py`

- [ ] **Step 1: Append failing tests to `tests/test_jotform_helper.py`**

Append these tests to the EXISTING `TestGetSubmissionCount` class (NOT a new class — add them as additional methods inside the existing class):

```python
    @patch("jotform_helper._suggest_form")
    @patch("jotform_helper._find_forms_by_name")
    def test_no_match_with_suggestion_returns_suggest(self, mock_find, mock_suggest):
        mock_find.return_value = []
        mock_suggest.return_value = _form("1", "Labour day Lower body Strength session", 40)
        result = jotform_helper.get_submission_count("lower body")
        self.assertEqual(result, {
            "status": "suggest",
            "name": "lower body",
            "suggestion": {
                "title": "Labour day Lower body Strength session",
                "count": 40,
            },
        })

    @patch("jotform_helper._suggest_form")
    @patch("jotform_helper._find_forms_by_name")
    def test_no_match_no_suggestion_returns_none(self, mock_find, mock_suggest):
        mock_find.return_value = []
        mock_suggest.return_value = None
        result = jotform_helper.get_submission_count("kettlebell")
        self.assertEqual(result, {"status": "none", "name": "kettlebell"})

    @patch("jotform_helper._suggest_form")
    @patch("jotform_helper._find_forms_by_name")
    def test_suggestion_count_is_coerced_to_int(self, mock_find, mock_suggest):
        mock_find.return_value = []
        # JotForm returns count as a string; suggestion count should also be coerced.
        mock_suggest.return_value = _form("1", "Lead Form", "99")
        result = jotform_helper.get_submission_count("lead")
        self.assertEqual(result["suggestion"]["count"], 99)
```

Also: the existing test `test_no_match_returns_none_status` (in the same `TestGetSubmissionCount` class) currently does NOT mock `_suggest_form`. After this task, the no-match branch will call `_suggest_form` against the live function (which would call `_api_get` against the actually-patched `_find_forms_by_name`... but `_suggest_form` calls `_api_get` directly, bypassing `_find_forms_by_name`). To keep the existing test isolated, modify it to also patch `_suggest_form` to return None.

Find the existing test:

```python
    @patch("jotform_helper._find_forms_by_name")
    def test_no_match_returns_none_status(self, mock_find):
        mock_find.return_value = []
        result = jotform_helper.get_submission_count("nope")
        self.assertEqual(result, {"status": "none", "name": "nope"})
```

Replace it with:

```python
    @patch("jotform_helper._suggest_form")
    @patch("jotform_helper._find_forms_by_name")
    def test_no_match_returns_none_status(self, mock_find, mock_suggest):
        mock_find.return_value = []
        mock_suggest.return_value = None
        result = jotform_helper.get_submission_count("nope")
        self.assertEqual(result, {"status": "none", "name": "nope"})
```

- [ ] **Step 2: Run tests to verify the new ones fail and the modified existing one still passes**

Run: `python -m unittest tests.test_jotform_helper.TestGetSubmissionCount -v`

Expected: 3 NEW test methods FAIL (`AssertionError` because `get_submission_count` still returns `{"status": "none", ...}` for the no-match case — it doesn't yet know about suggestions). The modified `test_no_match_returns_none_status` should still pass (since the no-suggestion path still returns `none`).

If the modified existing test fails, fix the patching before continuing.

- [ ] **Step 3: Update `get_submission_count` to call `_suggest_form` on no-match**

In `jotform_helper.py`, find the existing `if not matches:` line inside `get_submission_count`:

```python
    if not matches:
        return {"status": "none", "name": name}
```

Replace it with:

```python
    if not matches:
        suggestion = _suggest_form(name)
        if suggestion:
            return {
                "status": "suggest",
                "name": name,
                "suggestion": {
                    "title": suggestion.get("title", ""),
                    "count": int(suggestion.get("count", 0) or 0),
                },
            }
        return {"status": "none", "name": name}
```

Note: this new call MUST be inside the existing `try:` block so any `RuntimeError` / `requests.HTTPError` / `requests.RequestException` raised by `_suggest_form`'s `_api_get` call is caught by the existing exception handlers and converted to `{"status": "error", ...}`. Read the surrounding code to confirm the `if not matches:` block is inside the try — if so, no restructuring needed; just the substitution above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_jotform_helper.TestGetSubmissionCount -v`
Expected: 10 tests in this class PASS (7 existing — including the modified one — + 3 new).

Then full module: `python -m unittest tests.test_jotform_helper -v`
Expected: 33 tests total (30 + 3), all PASS.

- [ ] **Step 5: Commit**

```bash
git add jotform_helper.py tests/test_jotform_helper.py
git commit -m "feat: suggest closest match when no substring match found"
```

---

## Task 4: Add `suggest` branch in `app.py` handler

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the new branch to the handler**

In `/Users/samduncan/Desktop/claude-mcp/app.py`, find the `elif tool_name == "get_jotform_submissions":` block. Inside it, find this block:

```python
        elif result["status"] == "none":
            return f"No JotForm form found matching '{result['name']}'."
```

Immediately AFTER that block (and BEFORE the `elif result["status"] == "multiple":` block), add:

```python
        elif result["status"] == "suggest":
            s = result["suggestion"]
            return (
                f"No exact match for '{result['name']}'. "
                f"Did you mean *{s['title']}*? ({s['count']} submissions)"
            )
```

- [ ] **Step 2: Verify the file still parses**

Run: `python -c "import ast; ast.parse(open('app.py').read())"`
Expected: no output.

- [ ] **Step 3: Verify the wiring still works**

Run:

```bash
python -c "
import app
tools = app._get_tools_for_user('sam@hiitaustralia.com.au', '+61420233508')
assert any(t['name'] == 'get_jotform_submissions' for t in tools)
print('ok')
"
```
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: format suggest reply for closest-match fallback"
```

---

## Task 5: Local end-to-end smoke test (optional)

**Files:** none modified.

Verify the new path works against the live JotForm API. Skip if you'd rather smoke-test on Railway.

- [ ] **Step 1: Test a query designed to hit the suggest path**

The suggest path only fires when the query is NOT a substring of any active form's title BUT shares at least one meaningful token (>2 chars) with one. Pick a multi-word query whose words appear in a known form title but in a different order.

Run (replace `YOUR_KEY` with the JotForm API key):

```bash
JOTFORM_API_KEY=YOUR_KEY python -c "
from jotform_helper import get_submission_count
import json
# Multi-word query designed to hit token-overlap (no substring match in any form title).
# Pick words you know exist in a JotForm title but in different order.
# Example: if 'Midway Challenge Seminar' exists, query 'seminar midway' will be:
#   - NOT a substring of that title (order is 'Midway ... Seminar')
#   - Shares tokens 'seminar' and 'midway' with it → suggest fires
print('--- token-only query (likely suggest) ---')
print(json.dumps(get_submission_count('seminar midway'), indent=2))
print()
print('--- gibberish (should be none) ---')
print(json.dumps(get_submission_count('xyznonexistent'), indent=2))
"
```

Expected: the first call returns `{"status": "suggest", ...}` with a sensible title and count; the second returns `{"status": "none", ...}`.

If the first call returns `none`, your account doesn't have a form whose title contains both "seminar" and "midway" — substitute two words that DO appear (in different order) in a known form title and re-run.

No commit — verification only.

---

## Task 6: Deploy

**Files:** none modified.

- [ ] **Step 1: Confirm `JOTFORM_API_KEY` is still set on Railway**

Run from `/Users/samduncan/Desktop/claude-mcp`: `railway variables 2>&1 | grep -i JOTFORM`
Expected: a line showing `JOTFORM_API_KEY` is set. (No need to redact — the value is the user's; this is their machine.)

If missing, set it: `railway variables --set "JOTFORM_API_KEY=<value>"`.

- [ ] **Step 2: Deploy**

Run: `railway up --detach`
Expected: build kicks off, returns immediately. Wait ~60-90s for it to come up.

- [ ] **Step 3: Confirm new container is running**

Run: `railway logs --deployment --lines 20`
Expected: see fresh `gunicorn ... Booting worker` lines from a recent timestamp.

- [ ] **Step 4: WhatsApp smoke test**

Use the same query that worked in Task 5's smoke test (e.g. a multi-word query with token overlap but no substring match), e.g.:

> "How many submissions for seminar midway?"

Expected reply: `No exact match for 'seminar midway'. Did you mean *<closest form title>*? (N submissions)`

Then test the no-suggestion fallback:

> "How many submissions for xyznonexistent?"

Expected reply: `No JotForm form found matching 'xyznonexistent'.`

No commit — deployment only.

---

## Done

Spec requirements implemented:

- ✅ `_fuzzy_score` copied verbatim from `trello_helper.py`
- ✅ `_suggest_form` returns single best candidate above `MIN_SUGGESTION_SCORE`, filters DELETED, returns None for empty input
- ✅ `get_submission_count` returns new `{"status": "suggest", ...}` shape on no-substring-match-but-has-suggestion
- ✅ Errors from `_suggest_form` are caught by the same try/except as `_find_forms_by_name`
- ✅ `app.py` handler formats the suggest reply with form title and count
- ✅ Tests cover fuzzy scoring, suggestion logic, threshold gating, DELETED filtering, empty input, and the get_submission_count integration
- ✅ Existing tests still pass (no regressions)
