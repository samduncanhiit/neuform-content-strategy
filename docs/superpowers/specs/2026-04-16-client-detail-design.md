# Client Detail Tool — Design

**Date:** 2026-04-16
**Status:** Draft — awaiting user review
**Project:** HIIT Station Capalaba WhatsApp Bot

## Purpose

Give gym staff a single WhatsApp command to pull a member's profile:
current membership, how long they've been a member, and class attendance
with automatic 30-day / 90-day / all-time breakdown. Useful for quick
check-ins, retention conversations, and performance reviews.

## Requirements

1. One new tool: `get_client_detail(client_name, days_back=None)`.
2. Searches for the client by name internally (reuses existing
   `search_clients` logic). Returns a disambiguation list if multiple
   matches, "No client found" if zero.
3. For a unique match, returns:
   - Client full name
   - "Member since" date — the client's `CreationDate` from MindBody
     (when they first walked in the door, regardless of membership type)
   - Current active membership(s) — from MindBody
     `client/activeclientmemberships`. Only currently active contracts;
     historical/expired/terminated contracts are excluded.
   - Classes attended (signed-in visits only, excluding no-shows and
     late cancellations):
     - All time (lifetime total)
     - Last 30 days
     - Last 90 days
     - If `days_back` is provided, also show that custom window
4. Plain text output (no `*` bold, no decoration), matching the
   membership report style.
5. Claude returns the tool output verbatim — the existing CRITICAL RULE
   in SYSTEM_PROMPT covers all tool output, but the new tool's
   `SYSTEM_PROMPT` guidance should also mention it.
6. Available to all three users (Sam, Chonnie, Erin) — no password gate.

## Non-goals

- Historical membership list (only current active shown).
- Revenue or payment history per client.
- Booking history (only attended/signed-in visits count).
- Bulk client reports (this is one client at a time).

## Architecture

```
WhatsApp message ("how many classes has Jane done?")
  → app.py /webhook
  → Claude (tool use)
  → handle_tool_call("get_client_detail", {client_name, days_back})
  → mindbody_helper.get_client_detail(client_name, days_back)
      ├─ search_clients(client_name) → client list
      ├─ if 0 matches → "No client found for '{client_name}'"
      ├─ if 2+ matches → disambiguation list
      └─ if 1 match:
          ├─ client["CreationDate"] → member since
          ├─ client/activeclientmemberships?ClientId=X → active contracts
          └─ client/clientvisits?ClientId=X → visit history
              └─ filter to signed-in only, count per window
  → mindbody_helper.format_client_detail(result)
  → Twilio REST reply
```

## Components

### `mindbody_helper.get_client_detail(client_name, days_back=None)`

Searches for the client, fetches membership + visit data, returns a
result dict.

**Return value (single match):**

```python
{
    "status": "found",
    "name": "Jane Smith",
    "member_since": "2024-03-14",
    "memberships": ["All Access 6 Month"],
    "classes_all_time": 247,
    "classes_30d": 18,
    "classes_90d": 52,
    "classes_custom": 6,        # only present if days_back was given
    "classes_custom_label": 14,  # the days_back value, for display
}
```

**Return value (no match):**

```python
{
    "status": "not_found",
    "search_text": "Janee Smyth",
}
```

**Return value (multiple matches):**

```python
{
    "status": "multiple",
    "matches": [
        {"name": "Jane Smith", "id": 12345, "email": "jane@..."},
        {"name": "Jane Doe", "id": 12346, "email": "jdoe@..."},
    ],
}
```

**Implementation notes:**

- Reuses `search_clients(client_name)` from the existing codebase to
  find the client. `search_clients` already calls the MindBody
  `client/clients` endpoint with a `SearchText` parameter.
- For active memberships: calls `_api_get("client/activeclientmemberships",
  {"ClientIds": client_id})`. Extracts the `Name` field from each
  membership in the response.
- For visit history: calls `_paginated_get("client/clientvisits",
  "Visits", {"ClientId": client_id})` to get all visits. Filters to
  visits where the `SignedIn` field is `True` (or equivalent MindBody
  status indicating actual attendance). Counts visits falling within
  each window (all time, last 30 days, last 90 days, optional custom).
- Dates computed from `_now()` (AEST) matching the rest of the codebase.
- Cached for 1 hour per `(client_id, days_back)` pair using the
  existing `_cache_get`/`_cache_set` pattern.
- Single-client API calls are fast (~1-3 seconds total for search +
  memberships + visits), so no need for slow-ack keywords.

### `mindbody_helper.format_client_detail(result)`

Pure formatting function. Three modes based on `result["status"]`:

**Found:**

```
Jane Smith
Member since: 14 March 2024
Current membership: All Access 6 Month

Classes attended: 247 (all time)
  Last 30 days: 18
  Last 90 days: 52
```

If `days_back` was specified (e.g. 14):

```
Jane Smith
Member since: 14 March 2024
Current membership: All Access 6 Month

Classes attended: 247 (all time)
  Last 14 days: 6
  Last 30 days: 18
  Last 90 days: 52
```

If multiple active memberships:

```
Jane Smith
Member since: 14 March 2024
Current memberships:
  All Access 6 Month
  Student Membership

Classes attended: 247 (all time)
  Last 30 days: 18
  Last 90 days: 52
```

**Not found:**

```
No client found for "Janee Smyth". Check the spelling or try a
different name/email/phone.
```

**Multiple matches:**

```
Found 3 matches for "Jane":
1. Jane Smith (jane@email.com)
2. Jane Doe (jdoe@email.com)
3. Jane Brown (jbrown@email.com)

Which one did you mean?
```

### `app.py` wiring

1. New tool schema entry in `_MINDBODY_TOOLS`:
   - `name`: `"get_client_detail"`
   - `description`: mentions membership info, how long they've been a
     member, and class attendance counts. Tells Claude to use it when
     the user asks about a specific member's details, classes done, or
     membership.
   - `input_schema`: `client_name` (string, required) and `days_back`
     (integer, optional).
2. New routing in `handle_tool_call`: lazy import, call
   `get_client_detail(client_name, days_back)`, then
   `format_client_detail(result)`, return string.
3. `SYSTEM_PROMPT` addition: one line telling Claude when to use this
   tool and that it covers member details, membership info, and class
   attendance. Explicitly distinguishes from `search_clients` (which
   just finds a member) and `get_member_stats` (which is aggregate
   stats, not per-client).

## Edge cases

- **Client with zero visits:** report "Classes attended: 0 (all time)"
  with zeroes for all windows.
- **No active membership** (expired/suspended/terminated client): show
  "No active membership" instead of a contract name. Still show member
  since and class counts — the user may be investigating a lapsed member.
- **Visit history API fails:** report what we have (name, member since,
  memberships) and note "Visit data unavailable" in place of class counts.
- **Multiple active memberships:** list all on separate lines (rare but
  possible if MindBody has overlapping contracts).
- **`days_back` = 0 or negative:** ignore, show default windows only.
- **Very old members with huge visit history:** pagination via
  `_paginated_get` handles this (the same pattern used for classes and
  clients elsewhere in the codebase). Cap at `max_pages=20` to avoid
  runaway queries.

## Testing

- **Unit-testable pure functions:**
  - `format_client_detail` in all three modes (found / not_found /
    multiple) with synthetic fixtures.
  - Custom `days_back` rendering.
  - Edge cases: zero visits, no active membership, multiple memberships.
- **Integration smoke test (manual, post-deploy):**
  1. Send: "how many classes has [known member] done?" → verify output
     format, all three windows shown.
  2. Send: "give me [name]'s details for the last 2 weeks" → verify
     custom days_back window appears.
  3. Search for an ambiguous name → verify disambiguation list.
  4. Search for a non-existent name → verify "No client found" message.
  5. Cross-check one member's class count against MindBody web UI.
- **No mocked MindBody tests** — matches existing repo convention.
