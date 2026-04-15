# Membership Movement Report — Design

**Date:** 2026-04-16
**Status:** Draft — awaiting user review
**Project:** HIIT Station Capalaba WhatsApp Bot

## Purpose

Give Sam, Chonnie, and Erin an on-demand WhatsApp report covering signups and
cancellations of **debiting memberships** over a variable time window (default
3 months). "Debiting memberships" means the real paid contracts in
`TRACKED_MEMBERSHIPS` — explicitly excluding offers, casual passes, and
challenge memberships.

The existing `get_member_stats` tool already reports cancellations but is
hard-coded to 7 days and doesn't break signups down by membership type. This
tool fills the longer-horizon gap.

## Requirements

1. One combined tool that returns signups **and** cancellations in a single
   response.
2. Parameterized time window: Claude extracts `days_back` from natural-language
   messages (`"last month"` → 30, `"last 3 months"` → 90, `"last 6 months"` →
   180). Default 90.
3. Optional monthly breakdown when the user asks for one (`"broken down by
   month"`, `"month by month"`, etc.).
4. Only tracked memberships count — filtered via the existing
   `_is_tracked_membership()` helper so `TRACKED_MEMBERSHIPS` stays the single
   source of truth.
5. "Signup" = a tracked-membership contract whose `StartDate` falls in the
   window. "Cancellation" = a tracked-membership contract whose
   `TerminationDate` falls in the window. Client age is irrelevant — a
   long-standing casual-pass client who converts is a signup; a member whose
   contract ends is a cancellation.
6. Output grouped by membership type with counts, with names listed underneath
   each group.
7. Report must respect the existing 1500-char WhatsApp reply limit.
8. Available to all three users — no password gate (this isn't revenue data).

## Non-goals

- Real-time data. A cache TTL of ~1 hour is acceptable.
- Tracking upgrades/downgrades as a distinct event type. If a client upgrades
  mid-window they legitimately appear as a cancellation on the old contract
  and a signup on the new one — that's correct behaviour for this report.
- Revenue attribution. Money owed / collected is out of scope; this is a
  headcount report.
- Backfilling or correcting historic data.

## Architecture

```
WhatsApp message
   → app.py /webhook
   → Claude (tool use)
   → handle_tool_call("get_membership_movement", {days_back, split_by_month})
   → mindbody_helper.get_membership_movement()
       ├─ _get_all_clients_paginated(LastModifiedDate >= cutoff)
       ├─ for each client: client/clientcontracts
       │    └─ evaluate each contract's StartDate / TerminationDate
       └─ return {signups: [...], cancellations: [...], meta: {...}}
   → mindbody_helper.format_membership_movement(result, days_back, split_by_month)
   → Twilio REST reply
```

## Components

### `mindbody_helper.get_membership_movement(days_back=90, split_by_month=False)`

Pure data function. Returns a dict:

```python
{
    "days_back": 90,
    "window_start": "2026-01-16",
    "window_end":   "2026-04-16",
    "signups": [
        {"client_id": 1234, "contract_id": 5678, "name": "Jane Smith",
         "membership": "All Access 6 Month", "date": "2026-02-14"},
        ...
    ],
    "cancellations": [
        {"client_id": 2345, "contract_id": 6789, "name": "John Doe",
         "membership": "All Access 6 Month", "date": "2026-03-02"},
        ...
    ],
}
```

Implementation notes:
- Cached under key `membership_movement_{days_back}` for 1 hour (reuse
  `_cache_get`/`_cache_set` with `CACHE_TTL_CLASSES`). Cache is independent of
  `split_by_month` — formatting decides how to group at render time.
- Fetches candidate clients with
  `_get_all_clients_paginated({"LastModifiedDate": cutoff_iso, "IncludeInactive": "true"}, max_pages=15)`.
  The 15-page ceiling (~3000 clients) should comfortably cover 90-day windows
  at this gym; for a 180-day window we accept the same ceiling and truncate at
  the oldest edge if we ever hit it.
- For each candidate, calls `client/clientcontracts?ClientId=X` and iterates
  every contract. Skips any contract where `_is_tracked_membership(ContractName)`
  is false. For the rest:
  - If `StartDate[:10]` is within `[window_start, window_end]` inclusive →
    append to `signups`.
  - If `TerminationDate[:10]` is within `[window_start, window_end]` inclusive
    → append to `cancellations`.
- Dedupe inside each list on `(client_id, contract_id)` in case pagination
  returns a client twice.
- Skip contracts with blank `StartDate` (cannot attribute). Same for
  cancellations with blank `TerminationDate`.
- Sorts each list by `date` descending before returning.

### `mindbody_helper.format_membership_movement(result, days_back, split_by_month)`

Pure presentation function. Two modes.

**Flat mode** (default):

```
*Membership Movement — Last 90 Days*

*SIGNUPS: 12*
• All Access 6 Month — 5
   - Jane Smith (2026-02-14)
   - ...
• Conversion Flexi — 4
   - ...
• Student Membership — 3
   - ...

*CANCELLATIONS: 7*
• All Access 6 Month — 3
   - John Doe (2026-03-02)
   - ...
• Conversion 12 Months — 2
   - ...

*Net: +5*
```

**Monthly mode** (`split_by_month=True`):

```
*Membership Movement — Last 90 Days (by month)*

*── February 2026 ──*
Signups: 4   Cancellations: 2   Net: +2
  Signups:
   • All Access 6 Month: Jane Smith, ...
   • Student Membership: ...
  Cancellations:
   • All Access 6 Month: John Doe, ...

*── March 2026 ──*
...

*── April 2026 (partial) ──*
...

*Totals — Signups: 12 · Cancellations: 7 · Net: +5*
```

Formatting rules:
- Groups inside each section sorted by count descending, ties broken
  alphabetically by membership name.
- Within a group, names sorted by date descending.
- Current calendar month labelled `(partial)` in monthly mode.
- If the formatted message would exceed ~1500 chars, truncate the largest
  buckets with `... and N more` (same pattern as `get_member_stats` — see
  `mindbody_helper.py:660`).

### `app.py` wiring

1. Add `get_membership_movement` to the tool schema list alongside other
   MindBody tools. Description tells Claude to use it for
   cancellation/signup reports over longer windows, and explicitly states that
   only debiting memberships count.
2. Parameters in the schema: `days_back` (integer, default 90) and
   `split_by_month` (boolean, default false).
3. Add routing in `handle_tool_call`: lazy-import `mindbody_helper`, call
   `get_membership_movement(**args)`, then `format_membership_movement(...)`,
   return the formatted string.
4. Update `SYSTEM_PROMPT` with a single line:
   *"For signup / cancellation reports over a month, quarter, or half-year,
   call `get_membership_movement`. Only debiting memberships count — offers,
   casual passes, and challenge memberships are excluded automatically. Pass
   `split_by_month=true` when the user asks for a monthly breakdown."*
5. Per-user tool filter: available to all three users.
6. Add slow-ack keywords to `SLOW_KEYWORDS`:
   `"membership movement"`, `"signups last"`, `"cancellations last"`,
   `"cancelled last"`. Avoid the overly-broad `"last 3 months"`.

## Edge cases

- **Contract in both lists**: allowed. Someone who signed up AND cancelled in
  the window shows up in both — that's two real events.
- **Multiple tracked contracts per client**: each contract evaluated
  independently. Upgrading from Flexi to 6-Month mid-window legitimately
  produces one cancellation plus one signup.
- **Blank `StartDate` or `TerminationDate`**: skipped — no date means no
  attribution.
- **Pagination overflow** (more than 3000 modified clients in the window):
  accept truncation at the oldest edge. Log a warning. If this ever fires in
  practice, reconsider the `max_pages` ceiling or migrate to a direct
  `sale/contracts` date-range query (not implemented here because the v6
  endpoint isn't confirmed to include termination data).
- **MindBody API failure on `client/clientcontracts`**: wrap in try/except
  matching `_get_client_membership_info` — log warning, skip that client, keep
  going. Partial results are better than a hard failure.
- **New debiting membership type launched**: handled by the existing
  `TRACKED_MEMBERSHIPS` list — one edit there and every report picks it up.

## Testing

- **Unit-testable pure functions**:
  - Month-bucketing logic (given a list of events + window, produce correct
    buckets including `partial` flag on current month).
  - `format_membership_movement` in both modes with synthetic fixtures —
    verify grouping, sort order, truncation at 1500 chars.
- **Integration smoke test** (manual, post-deploy):
  1. Send WhatsApp: *"give me membership movement last 90 days"* → verify
     flat-mode output.
  2. Send WhatsApp: *"and broken down by month"* → verify monthly mode.
  3. Send WhatsApp: *"cancellations last month"* → verify Claude picks
     `days_back=30`.
  4. Check a known recent signup and a known recent cancellation both appear
     under the correct buckets.
- No mocked MindBody tests — matches existing repo convention.

## Open questions

None.

## Rollout

1. Implement `get_membership_movement` + `format_membership_movement` in
   `mindbody_helper.py`.
2. Wire into `app.py` (tool schema, routing, system prompt, slow keywords).
3. Commit + `railway up --detach`.
4. Run the integration smoke test via WhatsApp.
5. If slow (> 20s first call) even with cache + slow-ack, revisit by
   investigating the MindBody `sale/contracts` date-range endpoint as a
   single-query alternative to the per-client contract walk.
