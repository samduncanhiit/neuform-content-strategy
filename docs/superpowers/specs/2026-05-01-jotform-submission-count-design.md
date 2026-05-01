# JotForm Submission Count Tool — Design

**Date:** 2026-05-01
**Status:** Approved

## Summary

Add a WhatsApp bot tool that returns the total submission count for a JotForm form, looked up by form name at request time. Available to all users (Sam, Chonnie, Erin).

## Motivation

The bot already integrates MindBody, Outlook, Gmail, Google Calendar, and Trello. JotForm is the next system the team wants visibility into via WhatsApp. The first ask is simple: "how many submissions does form X have?" — total count only, no date filtering.

## Scope

### In scope
- One new bot tool: `get_jotform_submissions`
- Lookup by form name (case-insensitive)
- Return total all-time submission count for the matched form
- Handle "no match" and "multiple matches" cases gracefully
- Make available to all three users

### Out of scope (deferred)
- Date-range filtering (today / this week / custom)
- Listing all forms with counts
- Per-question / per-field submission breakdowns
- Reading individual submission content

## Architecture

Follow the same pattern as `trello_helper.py` and `gmail_helper.py`:

1. **New file:** `jotform_helper.py` — wraps the JotForm REST API (`https://api.jotform.com`).
2. **New tool def + handler in `app.py`** — lazy-imported inside `handle_tool_call`.
3. **New env var:** `JOTFORM_API_KEY`, set in Railway dashboard.
4. **`SYSTEM_PROMPT` updated** so Claude knows to call this tool for submission-count questions.

## Components

### `jotform_helper.py`

```
_api_get(path, params=None)
    Internal helper. GET https://api.jotform.com{path}?apiKey=...
    Returns parsed JSON. Raises on non-200.

_find_forms_by_name(name)
    GET /user/forms (paginated if needed)
    Returns list of form dicts whose 'title' contains `name` (case-insensitive).
    Each dict includes 'id', 'title', 'count' (total submissions), 'status'.
    Filters out forms with status == 'DELETED'.

get_submission_count(name)
    Calls _find_forms_by_name(name).
    Returns one of:
      - {"status": "ok", "title": "...", "count": N}
      - {"status": "none", "name": name}
      - {"status": "multiple", "matches": [{"title": ..., "count": N}, ...]}
```

JotForm includes `count` (total submissions) directly on the form object returned
by `/user/forms`, so a single API call is sufficient — no second call to
`/form/{id}/submissions/count` needed.

### `app.py` changes

**Tool definition** (added to the tools list, near existing tool defs):
```python
{
    "name": "get_jotform_submissions",
    "description": "Get the total submission count for a JotForm form, looked up by form name. "
                   "Use this when the user asks 'how many submissions for X', 'submission count', "
                   "or similar. Matches form name case-insensitively. If multiple forms match, "
                   "the bot will list them so the user can pick.",
    "input_schema": {
        "type": "object",
        "properties": {
            "form_name": {
                "type": "string",
                "description": "Name (or partial name) of the JotForm form."
            }
        },
        "required": ["form_name"]
    }
}
```

**Handler branch** (added inside `handle_tool_call`):
```python
elif tool_name == "get_jotform_submissions":
    from jotform_helper import get_submission_count
    result = get_submission_count(tool_input["form_name"])
    if result["status"] == "ok":
        return f"*{result['title']}*: {result['count']} submissions"
    elif result["status"] == "none":
        return f"No JotForm form found matching '{result['name']}'."
    else:  # multiple
        lines = ["Multiple forms match — which one?"]
        for m in result["matches"]:
            lines.append(f"• {m['title']} ({m['count']} submissions)")
        return "\n".join(lines)
```

**SYSTEM_PROMPT update:** add one sentence telling the model to use
`get_jotform_submissions` for submission-count questions.

### Per-user tool filtering

The bot already filters tools per user. This tool should be added to the
"available to all users" set so Sam, Chonnie, and Erin can all call it.

## Error handling

- Missing `JOTFORM_API_KEY` env var → tool returns
  `"JotForm is not configured."` (do not crash the bot).
- API request fails (network / 5xx) → tool returns
  `"Couldn't reach JotForm right now — try again in a moment."`
- 401 from JotForm → tool returns
  `"JotForm API key is invalid — check the Railway env var."`

All errors are caught inside `get_submission_count` and returned as plain
strings, matching the pattern of other helpers in this repo.

## Match resolution

- **Exact case-insensitive match** on `title` — return immediately as `ok`.
- **No exact match → substring match** on `title` (case-insensitive).
  - 0 results → `none`
  - 1 result → `ok`
  - 2+ results → `multiple` (let the user pick)

This avoids accidentally summing counts across forms with similar names.

## Testing

- Unit test `_find_forms_by_name` with mocked `/user/forms` responses
  covering: exact match, single substring match, multiple matches, no match,
  and a DELETED form being filtered out.
- Unit test `get_submission_count` end-to-end with the same mock variants.
- Manual smoke test on Railway after deploy: send a WhatsApp message
  asking for submissions on a known form name.

## Deployment

1. Add `JOTFORM_API_KEY` to Railway env vars before deploying.
2. `railway up --detach` from the repo root (per existing project convention).
3. Smoke test via WhatsApp.

## Open questions

None — design approved.
