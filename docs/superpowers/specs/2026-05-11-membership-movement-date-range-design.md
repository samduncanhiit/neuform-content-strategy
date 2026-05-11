# Membership Movement — Specific Date Range

**Date:** 2026-05-11
**Status:** Draft — awaiting user review

## Problem

The WhatsApp bot can already report MindBody signups and cancellations through the `get_membership_movement` tool, but the only way to scope the window is `days_back` (a rolling N-day window ending today), and the formatter forces the output into calendar-month buckets. Users cannot ask things like "show me signups and cancellations between March 1 and April 18" or "what cancellations did we get in the first half of April" without falling back to a full-calendar-month view.

## Goal

Let the user request signups and cancellations for an arbitrary date range and get a single, focused reply for that exact window.

## Non-goals

- Adding date-range support to other reports (`get_revenue`, `get_new_members`, `get_payment_failures`, `get_weekly_summary`). Each has its own quirks (revenue is hard-coded Mon–Sun; weekly summary depends on it). Those are tracked separately if/when needed.
- Changing the set of tracked membership types or the definition of signup/cancellation.
- Replacing the existing rolling-window / monthly-bucket behaviour. That stays as-is for queries like "membership report last 3 months".

## User-facing behaviour

The user can ask the bot, in WhatsApp:

- "Show me signups and cancellations between March 1 and April 18"
- "What signups did we get from Jan 15 to Feb 28?"
- "Cancellations in March"
- "Membership movement from April 1 to today"

The model resolves natural-language dates to ISO `YYYY-MM-DD` (today's date is already in the system prompt) and calls `get_membership_movement` with `start_date` and/or `end_date`. The reply is a single block:

```
Membership Report
Mar 1 — Apr 18, 2026

SIGNUPS: 14
• All Access — 9
• Conversion 6mo — 5

CANCELLATIONS: 6
• All Access — 4
• Conversion 12mo — 2

Net: +8
```

Existing queries like "membership report last 3 months" are unchanged — they still pass `days_back` and still render with per-month buckets.

## Design

### Tool signature

`get_membership_movement(days_back=90, start_date=None, end_date=None)`

Window resolution:

1. If either `start_date` or `end_date` is provided → **range mode**.
   - Missing `end_date` defaults to today.
   - Missing `start_date` defaults to `end_date - 90 days`.
   - `days_back` is ignored in this mode.
2. Otherwise → **monthly mode** (current behaviour).
   - Window is `today - days_back` to `today`.

Both dates are inclusive: a contract with `StartDate == window_start` or `TerminationDate == window_end` counts.

### Validation

The tool returns a friendly error string (not an exception) when:

- Either date is provided but does not match `YYYY-MM-DD`.
- `start_date > end_date`.
- The range exceeds 365 days (matches the existing `days_back` max).

The error string is returned to the model as the tool result so the bot can relay it back to the user, e.g. `"Invalid date range — start_date must be on or before end_date."`

### MindBody query

Same `clients` endpoint, same `LastModifiedDate` filter, but the filter is `window_start` instead of `today - days_back`. Same pagination, same contract scan, same `_is_tracked_membership` check.

### Result shape

```python
{
  "mode": "range" | "monthly",
  "days_back": int | None,         # populated in monthly mode only
  "window_start": "YYYY-MM-DD",
  "window_end":   "YYYY-MM-DD",
  "signups": [...],
  "cancellations": [...],
}
```

The signup/cancellation list items are unchanged from today.

### Caching

Cache key changes from `f"membership_movement_{days_back}"` to `f"membership_movement_{window_start}_{window_end}"` so different ranges don't collide. TTL stays the same (1 hour, `CACHE_TTL_CLASSES`).

### Formatter

`format_membership_movement(result)` branches on `result["mode"]`:

- `"range"` → single-block format shown above.
  - Header line: `Membership Report`
  - Date line: `<Mar 1> — <Apr 18>, <year>` (use a friendly month-day format; if the range spans years, include both years).
  - `SIGNUPS: N` plus a counts-only block by membership type (reuse `_format_counts_block`).
  - `CANCELLATIONS: N` plus a counts-only block.
  - `Net: ±N`.
- `"monthly"` → unchanged, calls `_format_membership_movement_monthly` exactly as today.

### Tool schema (`app.py`)

Add to `get_membership_movement`'s `input_schema.properties`:

```json
"start_date": {"type": "string", "description": "Start of date range, YYYY-MM-DD. Optional. If provided (with or without end_date), days_back is ignored and the report is rendered as a single combined block for the range."},
"end_date":   {"type": "string", "description": "End of date range, YYYY-MM-DD. Optional. Defaults to today when start_date is given."}
```

### Tool description (`app.py`)

Append to the existing description:

> For queries with explicit dates ("between March 1 and April 18", "from Jan 15 to Feb 28", "in March", "cancellations last week"), pass `start_date` and/or `end_date` in YYYY-MM-DD. For rolling-window queries ("last 3 months"), keep using `days_back`.

### System prompt (`app.py`)

Add one sentence near the membership-report guidance:

> For date-range membership-movement queries, pass `start_date` and `end_date` in YYYY-MM-DD; today's date is given above.

### Tool handler (`app.py`)

The `get_membership_movement` branch in `handle_tool_call` reads `start_date` and `end_date` from `tool_input` (both optional) and passes them through. It picks the formatter path based on `result["mode"]`.

## Edge cases

- **End date in the future.** Allowed — the function just won't find contracts whose StartDate/TerminationDate is in the future, so the report shows zero in that tail. Not worth special-casing.
- **Same start and end date.** A one-day report. Valid. Returns whatever started or terminated on that exact date.
- **Range spanning years.** The date-line header includes both years (`Dec 15, 2025 — Jan 14, 2026`).
- **Stale cache after a contract is cancelled mid-hour.** Same as today — 1-hour TTL applies. Not changing.

## Testing

- Unit tests for window resolution:
  - both dates given → uses them
  - only `start_date` → end defaults to today
  - only `end_date` → start defaults to end−90d
  - neither → monthly mode with `days_back`
- Unit tests for validation: malformed date string, start > end, range > 365 days each return the expected error string and do not hit MindBody.
- Unit test for formatter: given a stub result with `mode="range"`, output starts with `Membership Report` and contains the date line, `SIGNUPS: N`, `CANCELLATIONS: N`, and `Net:` lines.
- Unit test that `mode="monthly"` formatter output is byte-identical to today's output (regression guard).
- Manual smoke test via the live WhatsApp bot after deploy: ask "signups and cancellations between [past Monday] and [today]" and confirm a single-block reply with the right header dates.

## Files touched

- `mindbody_helper.py` — `get_membership_movement`, `format_membership_movement`, plus a small helper for the range header.
- `app.py` — tool schema, tool description, system prompt sentence, tool handler.
- `tests/test_membership_movement.py` — extend the existing test file with the new range-mode and validation cases listed above.
