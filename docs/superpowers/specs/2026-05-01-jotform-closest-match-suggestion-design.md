# JotForm Closest-Match Suggestion — Design

**Date:** 2026-05-01
**Status:** Approved
**Builds on:** `2026-05-01-jotform-submission-count-design.md`

## Summary

When the WhatsApp bot can't find a JotForm form whose title contains the user's query, it should fall back to a fuzzy-match suggestion. If the closest form's title shares at least one meaningful token with the query, reply with the suggestion (and its submission count). Otherwise keep the current "no match" message.

## Motivation

The first iteration returns `No JotForm form found matching 'X'.` whenever there's no substring match. That's a dead end — the user has to retype with the exact wording. JotForm form titles tend to be long and specific (e.g. `8 Week Transformation Challenge Round 14 Southside`), so substring matching alone misses near-hits like a forgotten round number or word order.

## Scope

### In scope
- Add a fuzzy-score function to `jotform_helper.py` (copied from `trello_helper._fuzzy_score`)
- Add a `_suggest_form(name)` helper that returns the single best candidate above a threshold, or `None`
- Update `get_submission_count`: when `_find_forms_by_name` returns no matches, call `_suggest_form` before returning
- Add a new return shape: `{"status": "suggest", "name": str, "suggestion": {"title": str, "count": int}}`
- Update the `app.py` handler to format the suggest reply
- Unit tests for the new path

### Out of scope (deferred)
- Edit-distance / Levenshtein typo correction (current scoring is token-overlap only)
- Suggesting multiple candidates (top-3 list)
- Suggestions when there are *some* substring matches (only kicks in on the "none" path)
- Asking the bot to auto-pick a near-match without confirmation

## Architecture

Single new behavior added to the existing `jotform_helper.py` flow:

```
get_submission_count(name)
    matches = _find_forms_by_name(name)
    if matches:                                     # existing paths unchanged
        ...
    else:
        suggestion = _suggest_form(name)            # NEW
        if suggestion:
            return {"status": "suggest", ...}       # NEW
        return {"status": "none", "name": name}     # existing fallback
```

`_suggest_form` makes its own `_api_get("/user/forms")` call (no shared cache yet — the helper is stateless and called rarely; an extra request on the no-match path is fine for now).

## Components

### `jotform_helper._fuzzy_score(needle, haystack)`

Direct copy of the existing function in `trello_helper.py:144`. Same scoring rules:

- `0` = no match
- `1-9` = at least one shared meaningful token (>2 chars)
- `10+` = substring match (with length bonus)

Copying rather than importing keeps the modules independent. The function is small (~25 lines) and a shared utility module would be premature given only two callers.

### `jotform_helper._suggest_form(name)`

```
1. Fetch all non-DELETED forms via _api_get
2. Score each form's title against `name`
3. Return the highest-scoring form dict if its score >= MIN_SUGGESTION_SCORE
4. Otherwise return None
```

`MIN_SUGGESTION_SCORE = 1` — at least one shared meaningful token. Since `_find_forms_by_name` already handles substring matches (which would score >=10), this path only ever produces token-overlap suggestions (1-9 range). Setting the threshold at 1 means "show anything with shared vocabulary"; tune up if it suggests irrelevant forms.

### `jotform_helper.get_submission_count(name)` — updated

Insert one new branch before the existing `none` return:

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

The error-handling try/except in `get_submission_count` already wraps the `_find_forms_by_name` call. The new `_suggest_form` call should be inside the same `try` block so its API call's errors get the same treatment (missing key, 401, network).

### `app.py` handler — new branch

In the `get_jotform_submissions` handler, add a `suggest` case:

```python
elif result["status"] == "suggest":
    s = result["suggestion"]
    return f"No exact match for '{result['name']}'. Did you mean *{s['title']}*? ({s['count']} submissions)"
```

Reply format example:
> `No exact match for 'lower body strength'. Did you mean *Labour day Lower body Strength session*? (40 submissions)`

## Return shape additions

| Status | Shape |
|--------|-------|
| `ok` | `{"status": "ok", "title": str, "count": int}` (unchanged) |
| `none` | `{"status": "none", "name": str}` (unchanged — fallback when no suggestion either) |
| `multiple` | `{"status": "multiple", "matches": [...]}` (unchanged) |
| `error` | `{"status": "error", "message": str}` (unchanged) |
| **`suggest`** | **`{"status": "suggest", "name": str, "suggestion": {"title": str, "count": int}}` (NEW)** |

## Error handling

Unchanged. The new `_suggest_form` call is inside the same `try` block, so any `RuntimeError` / `HTTPError` / `RequestException` it raises is caught by the existing handlers and returned as `{"status": "error", "message": ...}`.

## Match resolution

The decision tree becomes:

1. Empty/blank query → `none` (no suggestion attempted)
2. Exact case-insensitive title match (one or more) → `ok` (single) or `multiple`
3. Substring match (one or more, no exact) → `ok` (single) or `multiple`
4. **No substring match → score all forms; if best ≥ threshold → `suggest`** (NEW)
5. Otherwise → `none`

## Testing

New tests in `tests/test_jotform_helper.py`:

- `_fuzzy_score`: copy the same test set used for `trello_helper._fuzzy_score` (exact match highest, substring beats non-match, case-insensitive, token overlap, empty needle returns 0, no overlap returns 0)
- `_suggest_form`:
  - Returns the highest-scoring form when at least one form scores ≥ threshold
  - Returns `None` when no form scores ≥ threshold
  - Filters out DELETED forms (don't suggest a deleted form)
  - Returns `None` for empty/blank query
- `get_submission_count`:
  - Returns `suggest` when `_find_forms_by_name` returns `[]` and `_suggest_form` returns a candidate
  - Returns `none` when `_find_forms_by_name` returns `[]` and `_suggest_form` returns `None`
  - Existing tests still pass (the suggest path doesn't change ok / multiple / error paths)

## Deployment

No new env vars. Same flow: commit → `railway up --detach`.

## Open questions

None — design approved.
