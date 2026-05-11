# Membership Movement — Date Range Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the WhatsApp bot's `get_membership_movement` tool accept arbitrary `start_date` / `end_date` and render the result as a single combined block, while preserving the existing rolling-window `days_back` + monthly-bucket behaviour.

**Architecture:** Add a small window-resolution helper inside `mindbody_helper.py` that normalises `(days_back, start_date, end_date)` into a single window plus a `mode` tag ("range" or "monthly"). `get_membership_movement` calls it, scans MindBody as today, and attaches `mode` to the result. The formatter branches on `mode` — range → single-block format, monthly → existing per-month renderer. `app.py` adds two optional tool params, updates the tool description and system prompt, and routes the new inputs through the handler.

**Tech Stack:** Python 3, Flask, `mindbody_helper.py` pure functions, `unittest` (file: `tests/test_membership_movement.py`), MindBody API v6, Anthropic SDK, Twilio WhatsApp.

**Spec:** `docs/superpowers/specs/2026-05-11-membership-movement-date-range-design.md`

---

## File Structure

**Modify only — no new files:**

- `mindbody_helper.py`
  - New private helper `_resolve_movement_window(days_back, start_date, end_date)` returning `(mode, window_start, window_end)`. Raises `ValueError` with a user-facing message on bad input.
  - New private helper `_format_range_header(window_start, window_end)` returning e.g. `"Mar 1 — Apr 18, 2026"` or `"Dec 15, 2025 — Jan 14, 2026"` when years differ.
  - `get_membership_movement` signature gains `start_date=None, end_date=None`. Body uses the resolution helper, updates the cache key, and adds `mode` to the result.
  - `format_membership_movement` branches on `result.get("mode")`. New private helper `_format_membership_movement_range(result, net_str)` for the single-block render.
- `app.py`
  - Tool schema for `get_membership_movement` gains `start_date` and `end_date` optional string fields.
  - Tool description expanded to tell the model when to use them.
  - `SYSTEM_PROMPT` adds one sentence about date-range usage.
  - `handle_tool_call`'s `get_membership_movement` branch reads the two new inputs, calls the helper, catches `ValueError` and returns its message verbatim.
- `tests/test_membership_movement.py`
  - New `TestResolveWindow` class for the resolution helper.
  - New `TestFormatRange` class for the single-block formatter.
  - Extend `TestRangeHeader` for the header helper.

---

## Task 1: Window-resolution helper

**Files:**
- Modify: `mindbody_helper.py` (add helper near the existing `get_membership_movement` block, around line 1098)
- Test: `tests/test_membership_movement.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_membership_movement.py` above the `if __name__ == "__main__":` line:

```python
class TestResolveWindow(unittest.TestCase):
    def test_neither_date_uses_days_back(self):
        mode, ws, we = mindbody_helper._resolve_movement_window(
            days_back=90, start_date=None, end_date=None,
            today_iso="2026-05-11",
        )
        self.assertEqual(mode, "monthly")
        self.assertEqual(we, "2026-05-11")
        self.assertEqual(ws, "2026-02-10")  # 90 days before 2026-05-11

    def test_both_dates_uses_range(self):
        mode, ws, we = mindbody_helper._resolve_movement_window(
            days_back=90, start_date="2026-03-01", end_date="2026-04-18",
            today_iso="2026-05-11",
        )
        self.assertEqual(mode, "range")
        self.assertEqual(ws, "2026-03-01")
        self.assertEqual(we, "2026-04-18")

    def test_only_start_date_defaults_end_to_today(self):
        mode, ws, we = mindbody_helper._resolve_movement_window(
            days_back=90, start_date="2026-03-01", end_date=None,
            today_iso="2026-05-11",
        )
        self.assertEqual(mode, "range")
        self.assertEqual(ws, "2026-03-01")
        self.assertEqual(we, "2026-05-11")

    def test_only_end_date_defaults_start_to_end_minus_90(self):
        mode, ws, we = mindbody_helper._resolve_movement_window(
            days_back=90, start_date=None, end_date="2026-04-18",
            today_iso="2026-05-11",
        )
        self.assertEqual(mode, "range")
        self.assertEqual(we, "2026-04-18")
        self.assertEqual(ws, "2026-01-18")  # 90 days before 2026-04-18

    def test_invalid_start_date_format_raises(self):
        with self.assertRaises(ValueError) as ctx:
            mindbody_helper._resolve_movement_window(
                days_back=90, start_date="03/01/2026", end_date=None,
                today_iso="2026-05-11",
            )
        self.assertIn("YYYY-MM-DD", str(ctx.exception))

    def test_invalid_end_date_format_raises(self):
        with self.assertRaises(ValueError) as ctx:
            mindbody_helper._resolve_movement_window(
                days_back=90, start_date=None, end_date="not-a-date",
                today_iso="2026-05-11",
            )
        self.assertIn("YYYY-MM-DD", str(ctx.exception))

    def test_start_after_end_raises(self):
        with self.assertRaises(ValueError) as ctx:
            mindbody_helper._resolve_movement_window(
                days_back=90, start_date="2026-04-18", end_date="2026-03-01",
                today_iso="2026-05-11",
            )
        self.assertIn("on or before", str(ctx.exception))

    def test_range_over_365_days_raises(self):
        with self.assertRaises(ValueError) as ctx:
            mindbody_helper._resolve_movement_window(
                days_back=90, start_date="2024-01-01", end_date="2026-04-18",
                today_iso="2026-05-11",
            )
        self.assertIn("365", str(ctx.exception))

    def test_same_start_and_end_allowed(self):
        mode, ws, we = mindbody_helper._resolve_movement_window(
            days_back=90, start_date="2026-04-18", end_date="2026-04-18",
            today_iso="2026-05-11",
        )
        self.assertEqual(mode, "range")
        self.assertEqual(ws, "2026-04-18")
        self.assertEqual(we, "2026-04-18")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_membership_movement.TestResolveWindow -v`
Expected: All 9 tests fail with `AttributeError: module 'mindbody_helper' has no attribute '_resolve_movement_window'`.

- [ ] **Step 3: Implement the helper**

Add to `mindbody_helper.py` immediately above the existing `def get_membership_movement(days_back=90):` (currently line 1098):

```python
def _resolve_movement_window(days_back, start_date, end_date, today_iso=None):
    """Resolve membership-movement window inputs to (mode, window_start, window_end).

    mode is "range" when either start_date or end_date is provided, else "monthly".
    Dates are inclusive ISO YYYY-MM-DD strings. Raises ValueError with a
    user-facing message on bad input.
    """
    from datetime import datetime as _dt, timedelta as _td

    def _parse(label, s):
        try:
            return _dt.strptime(s, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid {label} — expected YYYY-MM-DD, got {s!r}."
            )

    today = (
        _dt.strptime(today_iso, "%Y-%m-%d").date()
        if today_iso else _now().date()
    )

    if start_date is None and end_date is None:
        ws_date = today - _td(days=days_back)
        return "monthly", ws_date.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    end_d = _parse("end_date", end_date) if end_date else today
    start_d = _parse("start_date", start_date) if start_date else (end_d - _td(days=90))

    if start_d > end_d:
        raise ValueError(
            "Invalid date range — start_date must be on or before end_date."
        )
    if (end_d - start_d).days > 365:
        raise ValueError(
            "Invalid date range — span must be 365 days or less."
        )

    return "range", start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")
```

Note: `_now()` is already defined in `mindbody_helper.py` (it returns a timezone-aware AEST `datetime`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_membership_movement.TestResolveWindow -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mindbody_helper.py tests/test_membership_movement.py
git commit -m "feat(mindbody): add _resolve_movement_window helper"
```

---

## Task 2: Range-header helper

**Files:**
- Modify: `mindbody_helper.py`
- Test: `tests/test_membership_movement.py`

- [ ] **Step 1: Write the failing tests**

Add a new test class to `tests/test_membership_movement.py` above the `if __name__ == "__main__":` line:

```python
class TestRangeHeader(unittest.TestCase):
    def test_same_year_no_year_on_left(self):
        out = mindbody_helper._format_range_header("2026-03-01", "2026-04-18")
        self.assertEqual(out, "Mar 1 — Apr 18, 2026")

    def test_same_month_same_year(self):
        out = mindbody_helper._format_range_header("2026-04-01", "2026-04-18")
        self.assertEqual(out, "Apr 1 — Apr 18, 2026")

    def test_same_day(self):
        out = mindbody_helper._format_range_header("2026-04-18", "2026-04-18")
        self.assertEqual(out, "Apr 18 — Apr 18, 2026")

    def test_cross_year_includes_both_years(self):
        out = mindbody_helper._format_range_header("2025-12-15", "2026-01-14")
        self.assertEqual(out, "Dec 15, 2025 — Jan 14, 2026")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_membership_movement.TestRangeHeader -v`
Expected: All 4 tests fail with `AttributeError: module 'mindbody_helper' has no attribute '_format_range_header'`.

- [ ] **Step 3: Implement the helper**

Add to `mindbody_helper.py` immediately above `_resolve_movement_window`:

```python
def _format_range_header(window_start, window_end):
    """Render a human-friendly inclusive date range, e.g. 'Mar 1 — Apr 18, 2026'.

    When the start and end fall in different years, both years are shown:
    'Dec 15, 2025 — Jan 14, 2026'.
    """
    from datetime import datetime as _dt

    ws = _dt.strptime(window_start, "%Y-%m-%d").date()
    we = _dt.strptime(window_end, "%Y-%m-%d").date()

    left_month = ws.strftime("%b")
    right_month = we.strftime("%b")
    left_day = ws.day
    right_day = we.day

    if ws.year == we.year:
        return f"{left_month} {left_day} — {right_month} {right_day}, {we.year}"
    return (
        f"{left_month} {left_day}, {ws.year} — "
        f"{right_month} {right_day}, {we.year}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_membership_movement.TestRangeHeader -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mindbody_helper.py tests/test_membership_movement.py
git commit -m "feat(mindbody): add _format_range_header helper"
```

---

## Task 3: Range-mode formatter

**Files:**
- Modify: `mindbody_helper.py`
- Test: `tests/test_membership_movement.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_membership_movement.py` above the `if __name__ == "__main__":` line:

```python
class TestFormatRange(unittest.TestCase):
    def _result(self):
        return {
            "mode": "range",
            "window_start": "2026-03-01",
            "window_end": "2026-04-18",
            "signups": [
                {"client_id": 1, "contract_id": 10, "name": "Jane Smith",
                 "membership": "All Access 6 Month", "date": "2026-03-14"},
                {"client_id": 2, "contract_id": 11, "name": "Alex Ng",
                 "membership": "All Access 6 Month", "date": "2026-04-01"},
                {"client_id": 3, "contract_id": 12, "name": "Sam Lee",
                 "membership": "Student Membership", "date": "2026-04-10"},
            ],
            "cancellations": [
                {"client_id": 4, "contract_id": 13, "name": "John Doe",
                 "membership": "All Access 6 Month", "date": "2026-03-22"},
            ],
        }

    def test_header_uses_range_dates_not_days_back(self):
        out = mindbody_helper.format_membership_movement(self._result())
        self.assertIn("Membership Report", out)
        self.assertIn("Mar 1 — Apr 18, 2026", out)
        self.assertNotIn("Last", out)  # no "Last N Days" header in range mode

    def test_counts_and_net(self):
        out = mindbody_helper.format_membership_movement(self._result())
        self.assertIn("SIGNUPS: 3", out)
        self.assertIn("CANCELLATIONS: 1", out)
        self.assertIn("Net: +2", out)

    def test_counts_by_membership_no_names(self):
        out = mindbody_helper.format_membership_movement(self._result())
        self.assertIn("All Access 6 Month — 2", out)
        self.assertIn("Student Membership — 1", out)
        self.assertNotIn("Jane Smith", out)
        self.assertNotIn("Sam Lee", out)
        self.assertNotIn("John Doe", out)

    def test_empty_range(self):
        out = mindbody_helper.format_membership_movement({
            "mode": "range",
            "window_start": "2026-04-01",
            "window_end": "2026-04-18",
            "signups": [], "cancellations": [],
        })
        self.assertIn("Apr 1 — Apr 18, 2026", out)
        self.assertIn("SIGNUPS: 0", out)
        self.assertIn("CANCELLATIONS: 0", out)
        self.assertIn("Net: 0", out)
        self.assertIn("(none)", out)

    def test_negative_net(self):
        result = self._result()
        result["signups"] = []
        out = mindbody_helper.format_membership_movement(result)
        self.assertIn("SIGNUPS: 0", out)
        self.assertIn("CANCELLATIONS: 1", out)
        self.assertIn("Net: -1", out)

    def test_respects_whatsapp_cap(self):
        result = {
            "mode": "range",
            "window_start": "2026-01-01",
            "window_end": "2026-04-18",
            "signups": [
                {"client_id": i, "contract_id": 1000 + i,
                 "name": f"Client {i}",
                 "membership": f"Test Plan {i:03d}",
                 "date": "2026-02-14"}
                for i in range(300)
            ],
            "cancellations": [],
        }
        out = mindbody_helper.format_membership_movement(result)
        self.assertLessEqual(len(out), mindbody_helper.WHATSAPP_MAX_CHARS)
```

Also extend the existing monthly-mode test to confirm it still works when called with no extra args (back-compat check). Add to `TestFormatMonthly`:

```python
    def test_monthly_mode_via_mode_field(self):
        result = self._result()
        result["mode"] = "monthly"
        result["days_back"] = 90
        out = mindbody_helper.format_membership_movement(result)
        self.assertIn("January 2026", out)
        self.assertIn("Last 90 Days", out)
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python -m unittest tests.test_membership_movement.TestFormatRange -v`
Expected: All 6 tests fail because `format_membership_movement` currently requires `days_back` to be passed positionally for the existing `Last N Days` header — i.e. either an arity error or missing-`mode` behaviour.

- [ ] **Step 3: Add the range formatter and update `format_membership_movement` to branch on `result["mode"]`**

In `mindbody_helper.py`, immediately above `_format_membership_movement_monthly` (currently line 1061), add:

```python
def _format_membership_movement_range(result, net_str):
    signups = result.get("signups", [])
    cancellations = result.get("cancellations", [])
    header = _format_range_header(result["window_start"], result["window_end"])

    lines = ["Membership Report", header, ""]
    lines.append(f"SIGNUPS: {len(signups)}")
    lines.extend(_format_counts_block(signups))
    lines.append("")
    lines.append(f"CANCELLATIONS: {len(cancellations)}")
    lines.extend(_format_counts_block(cancellations))
    lines.append("")
    lines.append(f"Net: {net_str}")

    return _truncate_to_whatsapp("\n".join(lines))
```

Then replace the body of `format_membership_movement` (currently lines 842–867) with:

```python
def format_membership_movement(result, days_back=None, split_by_month=True):
    """Format a membership movement result dict for WhatsApp.

    Branches on result["mode"]:
      - "range" → single combined block headed by the date range.
      - "monthly" (or absent) → existing per-month breakdown.

    The `days_back` and `split_by_month` parameters are kept for backwards
    compatibility with monthly-mode callers; they are ignored in range mode.
    """
    signups = result.get("signups", [])
    cancellations = result.get("cancellations", [])
    net = len(signups) - len(cancellations)
    net_str = f"+{net}" if net > 0 else str(net)

    if result.get("mode") == "range":
        return _format_membership_movement_range(result, net_str)

    effective_days_back = days_back if days_back is not None else result.get("days_back", 90)

    if split_by_month:
        return _format_membership_movement_monthly(result, effective_days_back, net_str)

    lines = [f"Membership Report — Last {effective_days_back} Days", ""]
    lines.append(f"SIGNUPS: {len(signups)}")
    lines.extend(_format_counts_block(signups))
    lines.append("")
    lines.append(f"CANCELLATIONS: {len(cancellations)}")
    lines.extend(_format_counts_block(cancellations))
    lines.append("")
    lines.append(f"Net: {net_str}")

    return _truncate_to_whatsapp("\n".join(lines))
```

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `python -m unittest tests.test_membership_movement -v`
Expected: All tests PASS, including the existing `TestFormatFlat`, `TestFormatMonthly`, and `TestTruncation` classes (regression guard for back-compat).

- [ ] **Step 5: Commit**

```bash
git add mindbody_helper.py tests/test_membership_movement.py
git commit -m "feat(mindbody): add range-mode formatter for membership movement"
```

---

## Task 4: Wire `start_date` / `end_date` into `get_membership_movement`

**Files:**
- Modify: `mindbody_helper.py` (function at lines 1098–1188)
- Test: `tests/test_membership_movement.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_membership_movement.py`:

```python
class TestGetMembershipMovementInputs(unittest.TestCase):
    """Behavioural tests that mock the MindBody fetch so we exercise the
    window-resolution + result-shape changes without hitting the API."""

    def setUp(self):
        # Reset any cached result between tests
        mindbody_helper._CACHE.clear() if hasattr(mindbody_helper, "_CACHE") else None

    def _patch_fetch(self, candidates=None, contracts_by_client=None):
        candidates = candidates or []
        contracts_by_client = contracts_by_client or {}

        def fake_paginated(*args, **kwargs):
            return candidates

        def fake_api_get(path, params=None):
            if path == "client/clientcontracts":
                cid = params.get("ClientId")
                return {"Contracts": contracts_by_client.get(cid, [])}
            return {}

        return fake_paginated, fake_api_get

    def test_default_call_returns_monthly_mode(self):
        fake_paginated, fake_api_get = self._patch_fetch()
        orig_p = mindbody_helper._get_all_clients_paginated
        orig_g = mindbody_helper._api_get
        mindbody_helper._get_all_clients_paginated = fake_paginated
        mindbody_helper._api_get = fake_api_get
        try:
            result = mindbody_helper.get_membership_movement()
        finally:
            mindbody_helper._get_all_clients_paginated = orig_p
            mindbody_helper._api_get = orig_g
        self.assertEqual(result["mode"], "monthly")
        self.assertEqual(result["days_back"], 90)

    def test_explicit_dates_return_range_mode(self):
        fake_paginated, fake_api_get = self._patch_fetch()
        orig_p = mindbody_helper._get_all_clients_paginated
        orig_g = mindbody_helper._api_get
        mindbody_helper._get_all_clients_paginated = fake_paginated
        mindbody_helper._api_get = fake_api_get
        try:
            result = mindbody_helper.get_membership_movement(
                start_date="2026-03-01", end_date="2026-04-18",
            )
        finally:
            mindbody_helper._get_all_clients_paginated = orig_p
            mindbody_helper._api_get = orig_g
        self.assertEqual(result["mode"], "range")
        self.assertEqual(result["window_start"], "2026-03-01")
        self.assertEqual(result["window_end"], "2026-04-18")

    def test_invalid_date_raises_valueerror(self):
        with self.assertRaises(ValueError):
            mindbody_helper.get_membership_movement(start_date="bogus")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_membership_movement.TestGetMembershipMovementInputs -v`
Expected: All 3 tests fail. The first two with `TypeError: get_membership_movement() got an unexpected keyword argument 'start_date'`; the third for the same reason.

- [ ] **Step 3: Update `get_membership_movement`**

Replace the body of `get_membership_movement` (currently lines 1098–1188) with:

```python
def get_membership_movement(days_back=90, start_date=None, end_date=None):
    """Return signups and cancellations of tracked memberships.

    Two modes:
      - Range mode: pass start_date and/or end_date (YYYY-MM-DD). Result["mode"]
        is "range" and result["days_back"] is None.
      - Monthly mode (default): pass days_back. Result["mode"] is "monthly" and
        the existing rolling-window / per-month renderer applies.

    Signup  = tracked contract with StartDate within the window.
    Cancel  = tracked contract with TerminationDate within the window.
    Only contracts matching TRACKED_MEMBERSHIPS count. A single contract can
    appear in both lists if it both started and terminated in the window.

    Cached for 1 hour per resolved (window_start, window_end). Raises
    ValueError with a user-facing message on bad date input.
    """
    mode, window_start, window_end = _resolve_movement_window(
        days_back=days_back, start_date=start_date, end_date=end_date,
    )

    cache_key = f"membership_movement_{window_start}_{window_end}"
    cached = _cache_get(cache_key, ttl=CACHE_TTL_CLASSES)
    if cached is not None:
        logger.info(
            f"Using cached membership movement ({window_start} → {window_end})"
        )
        return cached

    modified_since = f"{window_start}T00:00:00"

    candidates = _get_all_clients_paginated(
        {"LastModifiedDate": modified_since, "IncludeInactive": "true"},
        max_pages=15,
    )

    signups = []
    cancellations = []
    seen_signup = set()
    seen_cancellation = set()

    for c in candidates:
        client_id = c.get("Id")
        if not client_id:
            continue
        try:
            contract_data = _api_get("client/clientcontracts", {"ClientId": client_id})
        except Exception as e:
            logger.warning(f"clientcontracts lookup failed for {client_id}: {e}")
            continue

        for contract in contract_data.get("Contracts", []) or []:
            contract_name = contract.get("ContractName") or ""
            if not _is_tracked_membership(contract_name):
                continue

            contract_id = contract.get("Id") or contract.get("ContractId") or 0
            client_name = _client_name(c)

            start_d = (contract.get("StartDate") or "")[:10]
            if start_d and window_start <= start_d <= window_end:
                key = (client_id, contract_id)
                if key not in seen_signup:
                    seen_signup.add(key)
                    signups.append({
                        "client_id": client_id,
                        "contract_id": contract_id,
                        "name": client_name,
                        "membership": contract_name,
                        "date": start_d,
                    })

            term_date = (contract.get("TerminationDate") or "")[:10]
            if term_date and window_start <= term_date <= window_end:
                key = (client_id, contract_id)
                if key not in seen_cancellation:
                    seen_cancellation.add(key)
                    cancellations.append({
                        "client_id": client_id,
                        "contract_id": contract_id,
                        "name": client_name,
                        "membership": contract_name,
                        "date": term_date,
                    })

    signups.sort(key=lambda x: x["date"], reverse=True)
    cancellations.sort(key=lambda x: x["date"], reverse=True)

    result = {
        "mode": mode,
        "days_back": days_back if mode == "monthly" else None,
        "window_start": window_start,
        "window_end": window_end,
        "signups": signups,
        "cancellations": cancellations,
    }
    _cache_set(cache_key, result)
    logger.info(
        f"Membership movement [{mode}] {window_start} → {window_end}: "
        f"{len(signups)} signups, {len(cancellations)} cancellations "
        f"(scanned {len(candidates)} clients)"
    )
    return result
```

- [ ] **Step 4: Run the full test file**

Run: `python -m unittest tests.test_membership_movement -v`
Expected: ALL tests pass, including the new `TestGetMembershipMovementInputs` and every pre-existing class.

- [ ] **Step 5: Commit**

```bash
git add mindbody_helper.py tests/test_membership_movement.py
git commit -m "feat(mindbody): accept start_date/end_date in get_membership_movement"
```

---

## Task 5: Tool schema, handler, system prompt in `app.py`

**Files:**
- Modify: `app.py`
  - Tool schema: currently lines 349–369 (`get_membership_movement` entry)
  - Tool handler: currently lines 733–737
  - System prompt: currently lines 100–108

This task has no unit tests of its own — the wiring is exercised by hand and through the deployment smoke test in Task 6. Keep the changes tight and self-contained.

- [ ] **Step 1: Update the tool schema**

In `app.py`, replace the `get_membership_movement` entry (currently lines 349–369) with:

```python
    {
        "name": "get_membership_movement",
        "description": (
            "Membership report: signups AND cancellations of debiting memberships. "
            "Two ways to scope the window:\n"
            "  • For rolling windows ('membership report last 3 months', 'last 6 months', "
            "    'last month'), pass days_back. Output is broken down by calendar month "
            "    with counts per membership type.\n"
            "  • For specific date ranges ('between March 1 and April 18', 'in March', "
            "    'from Jan 15 to Feb 28', 'cancellations last week'), pass start_date "
            "    and/or end_date (YYYY-MM-DD). Output is a single combined block for "
            "    the range with counts per membership type. Today's date is given above; "
            "    resolve relative phrases to ISO dates before calling.\n"
            "ONLY debiting memberships count — casual passes, offers, and challenge "
            "memberships are excluded automatically. Use this whenever the user asks "
            "for a 'membership report' or any cancellation/signup breakdown."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days_back": {
                    "type": "integer",
                    "description": "Rolling window in days. 30=last month, 90=last 3 months, 180=last 6 months. Default 90, max 365. Ignored if start_date or end_date is provided.",
                    "default": 90,
                },
                "start_date": {
                    "type": "string",
                    "description": "Start of date range, YYYY-MM-DD. Optional. If provided (with or without end_date), days_back is ignored and the report is rendered as a single combined block for the range.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End of date range, YYYY-MM-DD. Optional. Defaults to today when start_date is given.",
                },
            },
            "required": [],
        },
    },
```

- [ ] **Step 2: Update the tool handler**

In `app.py`, replace the `get_membership_movement` handler branch (currently lines 733–737):

```python
    elif tool_name == "get_membership_movement":
        from mindbody_helper import get_membership_movement, format_membership_movement
        days = max(1, min(int(tool_input.get("days_back", 90) or 90), 365))
        result = get_membership_movement(days_back=days)
        return format_membership_movement(result, days_back=days, split_by_month=True)
```

with:

```python
    elif tool_name == "get_membership_movement":
        from mindbody_helper import get_membership_movement, format_membership_movement
        start_date = tool_input.get("start_date") or None
        end_date = tool_input.get("end_date") or None
        days = max(1, min(int(tool_input.get("days_back", 90) or 90), 365))
        try:
            result = get_membership_movement(
                days_back=days, start_date=start_date, end_date=end_date,
            )
        except ValueError as e:
            return str(e)
        return format_membership_movement(result, days_back=days, split_by_month=True)
```

- [ ] **Step 3: Update the system prompt**

In `app.py`, in the `SYSTEM_PROMPT` block (currently lines 100–108), find the existing membership-report guidance starting with `"IMPORTANT: When the user asks for a 'membership report', 'cancellations and signups',"` and replace the full block (from that line through `"active/suspended/expired snapshot only. "`) with:

```python
    "IMPORTANT: When the user asks for a 'membership report', 'cancellations and signups', "
    "or any signup/cancellation report, you MUST use get_membership_movement. "
    "For rolling windows ('membership report last 3 months', 'membership report for "
    "last month', 'last 6 months'), pass days_back; the output is broken down by "
    "calendar month with counts per membership type. For specific date ranges "
    "('between March 1 and April 18', 'in March', 'from Jan 15 to Feb 28', "
    "'cancellations last week'), pass start_date and/or end_date in YYYY-MM-DD "
    "(today's date is given below — resolve relative phrases first); the output is "
    "a single combined block for the range. Only debiting memberships count — "
    "casual passes, offers, and challenge memberships are excluded automatically. "
    "RETURN THE TOOL OUTPUT VERBATIM to the user — do NOT paraphrase, summarize, "
    "reformat, or drop sections. Do NOT use get_member_stats for these questions — "
    "that tool is for the current active/suspended/expired snapshot only. "
```

- [ ] **Step 4: Smoke check — Python syntax + imports load**

Run: `python -c "import app; print('OK')"`
Expected: `OK` printed; no import errors.

Run: `python -m unittest tests.test_membership_movement -v`
Expected: All tests still PASS (no regression from the wiring change).

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(app): expose start_date/end_date on get_membership_movement tool"
```

---

## Task 6: Manual smoke test on Railway

**Files:** none (deployment + WhatsApp test)

- [ ] **Step 1: Deploy to Railway**

Run: `railway up --detach`
Expected: build completes; Railway dashboard shows new deployment as active.

- [ ] **Step 2: Send WhatsApp message — range query**

From an authorised WhatsApp number, send: `"show me signups and cancellations between [date 7 days ago, written as a month/day phrase] and today"`.
Expected reply: starts with `Membership Report`, second line is a date range like `<Mon D> — <Mon D>, 2026`, has `SIGNUPS: N`, `CANCELLATIONS: N`, `Net: ±N`, and no per-month sub-sections.

- [ ] **Step 3: Send WhatsApp message — rolling-window query (regression)**

Send: `"membership report last 3 months"`.
Expected reply: existing format — header `Membership Report — Last 90 Days`, per-month sub-sections, `Totals — Signups: ... · Cancellations: ... · Net: ...` footer. Confirms back-compat.

- [ ] **Step 4: Send WhatsApp message — invalid range**

Send: `"signups between April 30 and April 1"` (deliberately reversed).
Expected reply: a single-line error like `Invalid date range — start_date must be on or before end_date.`

- [ ] **Step 5: If any reply is wrong, file a follow-up note and stop**

If the model picks `days_back` instead of the explicit dates for the range query, the tool description / system prompt likely needs sharper wording. If the formatter output is wrong, capture the exact text and revisit Task 3.

- [ ] **Step 6: Final marker commit (optional)**

If everything passes, no further commit is needed — the implementation is on the branch already. Open a PR if that fits the user's flow.

---

## Self-Review Notes

- **Spec coverage:** every numbered spec section maps to a task — window resolution (Task 1), validation (Task 1), range header (Task 2), formatter (Task 3), MindBody query / cache key / result shape (Task 4), tool schema / description / handler / system prompt (Task 5), manual smoke test (Task 6).
- **Type consistency:** the result dict's `mode`, `window_start`, `window_end`, `days_back`, `signups`, `cancellations` keys are referenced identically across `_resolve_movement_window` → `get_membership_movement` → `format_membership_movement` → `_format_membership_movement_range` → `_format_membership_movement_monthly`. The function name `_format_range_header` is used the same way in Task 2 (definition) and Task 3 (call site).
- **Placeholder scan:** no TBDs, no "add appropriate error handling", no implicit "similar to Task N". Each code step shows the full block to write.
- **Cache-key migration:** intentional. Old keys (`membership_movement_90`) become stale and age out — they're never read again because the new format is the only one written. No migration code needed.
