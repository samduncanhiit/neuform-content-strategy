# Membership Movement Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a WhatsApp-triggered report for signups and cancellations of debiting memberships over a variable window (default 90 days), with optional monthly breakdown.

**Architecture:** New `get_membership_movement()` data function and `format_membership_movement()` presentation function in `mindbody_helper.py`. Reuses existing helpers (`_get_all_clients_paginated`, `_is_tracked_membership`, `_cache_get`/`_cache_set`) and the per-client `client/clientcontracts` walk pattern already used by `_get_client_membership_info`. New tool entry + routing in `app.py`. Pure functions (formatter, month bucketer) covered by stdlib `unittest` tests; the live MindBody call is smoke-tested in production after deploy.

**Tech Stack:** Python 3, Flask, Anthropic SDK (tool use), MindBody Public API v6, stdlib `unittest` (no new deps).

**Spec:** `docs/superpowers/specs/2026-04-16-membership-movement-report-design.md`

---

## File Structure

- **Modify** `mindbody_helper.py`
  - Add `_bucket_movement_by_month(signups, cancellations, window_start, window_end)` — pure function
  - Add `get_membership_movement(days_back=90)` — data fetcher + cache
  - Add `format_membership_movement(result, days_back, split_by_month=False)` — formatter
- **Modify** `app.py`
  - New tool schema entry in `_MINDBODY_TOOLS` (around line 284, after `get_new_members`)
  - New routing branch in `handle_tool_call` (around line 524, after `get_new_members`)
  - One new line in `SYSTEM_PROMPT` (around line 74, in the tools-available paragraph)
- **Create** `tests/__init__.py` — empty file, enables `python -m unittest discover tests`
- **Create** `tests/test_membership_movement.py` — unit tests for pure functions

---

## Task 1: Scaffold the tests directory

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_membership_movement.py`

- [ ] **Step 1: Create the empty package marker**

Create `tests/__init__.py` with no content. This lets `python -m unittest discover` find the tests.

- [ ] **Step 2: Create a smoke test**

Create `tests/test_membership_movement.py`:

```python
"""Unit tests for the membership movement report pure functions."""
import unittest

import mindbody_helper


class TestScaffolding(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(hasattr(mindbody_helper, "_is_tracked_membership"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the smoke test to confirm discovery works**

Run from repo root: `python -m unittest discover tests -v`
Expected: `test_module_imports ... ok` and `OK` at the end.

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/test_membership_movement.py
git commit -m "Scaffold tests directory with smoke test for mindbody_helper"
```

---

## Task 2: Month-bucketing helper (TDD)

**Files:**
- Modify: `mindbody_helper.py` — add `_bucket_movement_by_month()` near the other helpers
- Modify: `tests/test_membership_movement.py` — add `TestBucketByMonth` class

The helper returns an ordered list of month buckets, each marked `partial` if the window cuts through that calendar month (i.e. the window does not cover the full 1st-to-last-day).

- [ ] **Step 1: Write failing tests for the bucketing logic**

Append to `tests/test_membership_movement.py`:

```python
class TestBucketByMonth(unittest.TestCase):
    def _event(self, date, name="X", membership="All Access 6 Month"):
        return {"client_id": 1, "contract_id": 1, "name": name,
                "membership": membership, "date": date}

    def test_events_grouped_by_calendar_month(self):
        signups = [
            self._event("2026-02-14", "Jane"),
            self._event("2026-03-02", "Bob"),
            self._event("2026-03-30", "Carol"),
        ]
        cancellations = [self._event("2026-02-20", "Dan")]
        buckets = mindbody_helper._bucket_movement_by_month(
            signups, cancellations,
            window_start="2026-01-16", window_end="2026-04-16",
        )
        # Expect four buckets: Jan, Feb, Mar, Apr — chronological
        labels = [b["month_label"] for b in buckets]
        self.assertEqual(labels,
            ["January 2026", "February 2026", "March 2026", "April 2026"])
        # Feb: 1 signup, 1 cancellation
        feb = buckets[1]
        self.assertEqual(len(feb["signups"]), 1)
        self.assertEqual(feb["signups"][0]["name"], "Jane")
        self.assertEqual(len(feb["cancellations"]), 1)
        # Mar: 2 signups
        self.assertEqual(len(buckets[2]["signups"]), 2)
        # Jan and Apr: empty
        self.assertEqual(buckets[0]["signups"], [])
        self.assertEqual(buckets[3]["signups"], [])

    def test_partial_flag_on_edge_months(self):
        buckets = mindbody_helper._bucket_movement_by_month(
            signups=[], cancellations=[],
            window_start="2026-01-16", window_end="2026-04-16",
        )
        partial_map = {b["month_label"]: b["partial"] for b in buckets}
        self.assertTrue(partial_map["January 2026"])   # starts mid-month
        self.assertFalse(partial_map["February 2026"]) # full month
        self.assertFalse(partial_map["March 2026"])    # full month
        self.assertTrue(partial_map["April 2026"])     # window ends mid-month

    def test_full_calendar_window_has_no_partials(self):
        buckets = mindbody_helper._bucket_movement_by_month(
            signups=[], cancellations=[],
            window_start="2026-02-01", window_end="2026-03-31",
        )
        self.assertFalse(buckets[0]["partial"])
        self.assertFalse(buckets[1]["partial"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_membership_movement.TestBucketByMonth -v`
Expected: `AttributeError: module 'mindbody_helper' has no attribute '_bucket_movement_by_month'` for all three tests.

- [ ] **Step 3: Implement the helper**

Add this function in `mindbody_helper.py` immediately after `_is_tracked_membership` (around line 326):

```python
def _bucket_movement_by_month(signups, cancellations, window_start, window_end):
    """Bucket a list of signup/cancellation events into calendar-month buckets.

    Returns an ordered list (oldest first) of dicts:
        {"month_label": "February 2026",
         "year_month":  "2026-02",
         "partial":     bool,
         "signups":     [event, ...],
         "cancellations": [event, ...]}

    A bucket is marked `partial` when the window does not cover the full
    calendar month (e.g. window_start > 1st of month, or window_end < last of month).
    """
    from calendar import monthrange

    def _ym(date_str):
        return date_str[:7]  # "YYYY-MM"

    def _label(ym):
        y, m = ym.split("-")
        names = ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]
        return f"{names[int(m) - 1]} {y}"

    # Build ordered list of months spanned by the window
    start_ym = _ym(window_start)
    end_ym = _ym(window_end)
    months = []
    y, m = int(start_ym[:4]), int(start_ym[5:7])
    ey, em = int(end_ym[:4]), int(end_ym[5:7])
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1

    buckets = []
    for ym in months:
        year, mon = int(ym[:4]), int(ym[5:7])
        first_day = f"{ym}-01"
        last_day = f"{ym}-{monthrange(year, mon)[1]:02d}"
        partial = window_start > first_day or window_end < last_day
        buckets.append({
            "month_label": _label(ym),
            "year_month": ym,
            "partial": partial,
            "signups": [e for e in signups if _ym(e["date"]) == ym],
            "cancellations": [e for e in cancellations if _ym(e["date"]) == ym],
        })
    return buckets
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_membership_movement.TestBucketByMonth -v`
Expected: three tests `ok`, final `OK`.

- [ ] **Step 5: Commit**

```bash
git add mindbody_helper.py tests/test_membership_movement.py
git commit -m "Add _bucket_movement_by_month helper with unit tests"
```

---

## Task 3: `format_membership_movement` — flat mode (TDD)

**Files:**
- Modify: `mindbody_helper.py` — add `format_membership_movement()` near the other formatters (e.g. after `format_new_members`)
- Modify: `tests/test_membership_movement.py` — add `TestFormatFlat` class

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_membership_movement.py`:

```python
class TestFormatFlat(unittest.TestCase):
    def _result(self):
        return {
            "days_back": 90,
            "window_start": "2026-01-16",
            "window_end": "2026-04-16",
            "signups": [
                {"client_id": 1, "contract_id": 10, "name": "Jane Smith",
                 "membership": "All Access 6 Month", "date": "2026-02-14"},
                {"client_id": 2, "contract_id": 11, "name": "Alex Ng",
                 "membership": "All Access 6 Month", "date": "2026-03-01"},
                {"client_id": 3, "contract_id": 12, "name": "Sam Lee",
                 "membership": "Student Membership", "date": "2026-02-20"},
            ],
            "cancellations": [
                {"client_id": 4, "contract_id": 13, "name": "John Doe",
                 "membership": "All Access 6 Month", "date": "2026-03-02"},
            ],
        }

    def test_header_and_counts(self):
        out = mindbody_helper.format_membership_movement(
            self._result(), days_back=90, split_by_month=False)
        self.assertIn("Membership Movement", out)
        self.assertIn("Last 90 Days", out)
        self.assertIn("SIGNUPS: 3", out)
        self.assertIn("CANCELLATIONS: 1", out)
        self.assertIn("Net: +2", out)

    def test_groups_by_membership_with_names(self):
        out = mindbody_helper.format_membership_movement(
            self._result(), days_back=90, split_by_month=False)
        # Biggest group first: All Access 6 Month has 2 signups
        self.assertIn("All Access 6 Month", out)
        self.assertIn("Jane Smith", out)
        self.assertIn("Alex Ng", out)
        self.assertIn("Student Membership", out)
        self.assertIn("Sam Lee", out)
        self.assertIn("John Doe", out)
        # Names have their dates alongside
        self.assertIn("2026-02-14", out)

    def test_empty_result(self):
        out = mindbody_helper.format_membership_movement(
            {"days_back": 30, "window_start": "2026-03-17", "window_end": "2026-04-16",
             "signups": [], "cancellations": []},
            days_back=30, split_by_month=False)
        self.assertIn("SIGNUPS: 0", out)
        self.assertIn("CANCELLATIONS: 0", out)
        self.assertIn("Net: 0", out)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_membership_movement.TestFormatFlat -v`
Expected: `AttributeError: module 'mindbody_helper' has no attribute 'format_membership_movement'`.

- [ ] **Step 3: Implement the formatter (flat mode only for now)**

Add this function in `mindbody_helper.py` after `format_new_members`:

```python
# ── Membership Movement Report ────────────────────────────────────────────────

WHATSAPP_MAX_CHARS = 1500


def format_membership_movement(result, days_back=90, split_by_month=False):
    """Format a membership movement result dict for WhatsApp.

    Flat mode: grouped by membership with names underneath.
    Monthly mode: same groupings bucketed into calendar months (see Task 4).
    """
    signups = result.get("signups", [])
    cancellations = result.get("cancellations", [])
    net = len(signups) - len(cancellations)
    net_str = f"+{net}" if net > 0 else str(net)

    if split_by_month:
        return _format_membership_movement_monthly(result, days_back, net_str)

    lines = [f"*Membership Movement — Last {days_back} Days*", ""]
    lines.append(f"*SIGNUPS: {len(signups)}*")
    lines.extend(_format_group_block(signups))
    lines.append("")
    lines.append(f"*CANCELLATIONS: {len(cancellations)}*")
    lines.extend(_format_group_block(cancellations))
    lines.append("")
    lines.append(f"*Net: {net_str}*")

    return _truncate_to_whatsapp("\n".join(lines))


def _format_group_block(events):
    """Group events by membership name, sort by count desc, render bullets."""
    if not events:
        return ["(none)"]
    by_mem = {}
    for e in events:
        by_mem.setdefault(e["membership"], []).append(e)
    # Sort groups: count desc, then membership name asc
    ordered = sorted(by_mem.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    out = []
    for mem_name, group in ordered:
        out.append(f"• {mem_name} — {len(group)}")
        # Names sorted by date desc
        for e in sorted(group, key=lambda x: x["date"], reverse=True):
            out.append(f"   - {e['name']} ({e['date']})")
    return out


def _truncate_to_whatsapp(text):
    """If text exceeds WHATSAPP_MAX_CHARS, truncate and append a notice."""
    if len(text) <= WHATSAPP_MAX_CHARS:
        return text
    cutoff = WHATSAPP_MAX_CHARS - 40
    return text[:cutoff].rstrip() + "\n… (truncated — ask for a month)"


def _format_membership_movement_monthly(result, days_back, net_str):
    # Stub — implemented in Task 4.
    raise NotImplementedError("monthly mode implemented in Task 4")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_membership_movement.TestFormatFlat -v`
Expected: three tests `ok`.

Run the full test file too: `python -m unittest tests.test_membership_movement -v`
Expected: all tests so far pass.

- [ ] **Step 5: Commit**

```bash
git add mindbody_helper.py tests/test_membership_movement.py
git commit -m "Add format_membership_movement flat mode with unit tests"
```

---

## Task 4: `format_membership_movement` — monthly mode (TDD)

**Files:**
- Modify: `mindbody_helper.py` — implement `_format_membership_movement_monthly()`
- Modify: `tests/test_membership_movement.py` — add `TestFormatMonthly` class

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_membership_movement.py`:

```python
class TestFormatMonthly(unittest.TestCase):
    def _result(self):
        return {
            "days_back": 90,
            "window_start": "2026-01-16",
            "window_end": "2026-04-16",
            "signups": [
                {"client_id": 1, "contract_id": 10, "name": "Jane Smith",
                 "membership": "All Access 6 Month", "date": "2026-02-14"},
                {"client_id": 2, "contract_id": 11, "name": "Alex Ng",
                 "membership": "Student Membership", "date": "2026-03-01"},
            ],
            "cancellations": [
                {"client_id": 3, "contract_id": 12, "name": "John Doe",
                 "membership": "All Access 6 Month", "date": "2026-03-02"},
                {"client_id": 4, "contract_id": 13, "name": "Kim Lee",
                 "membership": "All Access 6 Month", "date": "2026-04-05"},
            ],
        }

    def test_header_says_by_month(self):
        out = mindbody_helper.format_membership_movement(
            self._result(), days_back=90, split_by_month=True)
        self.assertIn("by month", out)
        self.assertIn("Last 90 Days", out)

    def test_each_month_section_present(self):
        out = mindbody_helper.format_membership_movement(
            self._result(), days_back=90, split_by_month=True)
        self.assertIn("January 2026", out)
        self.assertIn("February 2026", out)
        self.assertIn("March 2026", out)
        self.assertIn("April 2026", out)

    def test_partial_flag_rendered_on_edges(self):
        out = mindbody_helper.format_membership_movement(
            self._result(), days_back=90, split_by_month=True)
        # Both edge months are partial
        self.assertIn("January 2026 (partial)", out)
        self.assertIn("April 2026 (partial)", out)
        # Middle months are not
        self.assertNotIn("February 2026 (partial)", out)
        self.assertNotIn("March 2026 (partial)", out)

    def test_totals_footer(self):
        out = mindbody_helper.format_membership_movement(
            self._result(), days_back=90, split_by_month=True)
        self.assertIn("Signups: 2", out)
        self.assertIn("Cancellations: 2", out)
        self.assertIn("Net: 0", out)

    def test_events_appear_under_correct_month(self):
        out = mindbody_helper.format_membership_movement(
            self._result(), days_back=90, split_by_month=True)
        # Jane signed up in Feb
        feb_idx = out.find("February 2026")
        mar_idx = out.find("March 2026")
        apr_idx = out.find("April 2026")
        jane_idx = out.find("Jane Smith")
        kim_idx = out.find("Kim Lee")
        self.assertGreater(jane_idx, feb_idx)
        self.assertLess(jane_idx, mar_idx)
        self.assertGreater(kim_idx, apr_idx)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_membership_movement.TestFormatMonthly -v`
Expected: all five fail with `NotImplementedError: monthly mode implemented in Task 4`.

- [ ] **Step 3: Implement monthly mode**

Replace the `_format_membership_movement_monthly` stub in `mindbody_helper.py` with:

```python
def _format_membership_movement_monthly(result, days_back, net_str):
    signups = result.get("signups", [])
    cancellations = result.get("cancellations", [])
    buckets = _bucket_movement_by_month(
        signups, cancellations,
        window_start=result["window_start"],
        window_end=result["window_end"],
    )

    lines = [f"*Membership Movement — Last {days_back} Days (by month)*", ""]
    for b in buckets:
        label = b["month_label"] + (" (partial)" if b["partial"] else "")
        bucket_net = len(b["signups"]) - len(b["cancellations"])
        bucket_net_str = f"+{bucket_net}" if bucket_net > 0 else str(bucket_net)
        lines.append(f"*── {label} ──*")
        lines.append(
            f"Signups: {len(b['signups'])}   "
            f"Cancellations: {len(b['cancellations'])}   "
            f"Net: {bucket_net_str}"
        )
        if b["signups"]:
            lines.append("  Signups:")
            for sub in _summarise_by_membership(b["signups"]):
                lines.append(f"   • {sub}")
        if b["cancellations"]:
            lines.append("  Cancellations:")
            for sub in _summarise_by_membership(b["cancellations"]):
                lines.append(f"   • {sub}")
        lines.append("")

    lines.append(
        f"*Totals — Signups: {len(signups)} · "
        f"Cancellations: {len(cancellations)} · Net: {net_str}*"
    )
    return _truncate_to_whatsapp("\n".join(lines))


def _summarise_by_membership(events):
    """Render a compact one-line-per-membership summary: 'Name: A, B, C'."""
    by_mem = {}
    for e in events:
        by_mem.setdefault(e["membership"], []).append(e)
    ordered = sorted(by_mem.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    lines = []
    for mem_name, group in ordered:
        names = ", ".join(
            e["name"] for e in sorted(group, key=lambda x: x["date"], reverse=True)
        )
        lines.append(f"{mem_name}: {names}")
    return lines
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_membership_movement.TestFormatMonthly -v`
Expected: five tests `ok`.

Full file check: `python -m unittest tests.test_membership_movement -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add mindbody_helper.py tests/test_membership_movement.py
git commit -m "Add format_membership_movement monthly mode with unit tests"
```

---

## Task 5: Truncation safety for oversized results (TDD)

**Files:**
- Modify: `tests/test_membership_movement.py` — add `TestTruncation` class

Verify that both formatter modes stay within `WHATSAPP_MAX_CHARS` (1500) even when passed a pathologically large result. The truncation helper already exists (`_truncate_to_whatsapp` from Task 3); this task just pins the behaviour with a test.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_membership_movement.py`:

```python
class TestTruncation(unittest.TestCase):
    def _big_result(self, n=300):
        mems = ["All Access 6 Month", "Conversion Flexi", "Student Membership"]
        signups = [
            {"client_id": i, "contract_id": 1000 + i,
             "name": f"Client Number {i:03d}",
             "membership": mems[i % 3],
             "date": f"2026-02-{(i % 28) + 1:02d}"}
            for i in range(n)
        ]
        return {
            "days_back": 90,
            "window_start": "2026-01-16", "window_end": "2026-04-16",
            "signups": signups, "cancellations": [],
        }

    def test_flat_mode_respects_cap(self):
        out = mindbody_helper.format_membership_movement(
            self._big_result(), days_back=90, split_by_month=False)
        self.assertLessEqual(len(out), mindbody_helper.WHATSAPP_MAX_CHARS)

    def test_monthly_mode_respects_cap(self):
        out = mindbody_helper.format_membership_movement(
            self._big_result(), days_back=90, split_by_month=True)
        self.assertLessEqual(len(out), mindbody_helper.WHATSAPP_MAX_CHARS)

    def test_truncation_marker_present_when_truncated(self):
        out = mindbody_helper.format_membership_movement(
            self._big_result(), days_back=90, split_by_month=False)
        self.assertIn("truncated", out)
```

- [ ] **Step 2: Run the tests**

Run: `python -m unittest tests.test_membership_movement.TestTruncation -v`
Expected: PASS — `_truncate_to_whatsapp` from Task 3 already enforces the cap. If any test fails, fix the bug in `_truncate_to_whatsapp` before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_membership_movement.py
git commit -m "Pin WhatsApp length cap for membership movement formatter"
```

---

## Task 6: `get_membership_movement` data fetcher

**Files:**
- Modify: `mindbody_helper.py` — add `get_membership_movement()` near `get_new_members`

This is the live MindBody call. The repo convention is to not mock MindBody, so this function is not unit-tested — it is smoke-tested in production in Task 8. The function is small and composes existing helpers.

- [ ] **Step 1: Add the data fetcher**

Add this function in `mindbody_helper.py` immediately after `format_new_members` (and before the `# ── Arrears Report ──` section):

```python
# ── Membership Movement ───────────────────────────────────────────────────────


def get_membership_movement(days_back=90):
    """Return signups and cancellations of tracked memberships over the last N days.

    Signup  = tracked contract with StartDate within the window.
    Cancel  = tracked contract with TerminationDate within the window.
    Only contracts matching TRACKED_MEMBERSHIPS count. A single contract can
    appear in both lists if it both started and terminated in the window.

    Cached for 1 hour per days_back value.
    """
    cache_key = f"membership_movement_{days_back}"
    cached = _cache_get(cache_key, ttl=CACHE_TTL_CLASSES)
    if cached is not None:
        logger.info(f"Using cached membership movement ({days_back}d)")
        return cached

    window_end = _now().strftime("%Y-%m-%d")
    window_start = (_now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    modified_since = (_now() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00")

    # Fetch candidate clients — anyone touched in the window (new contracts and
    # terminations both bump LastModifiedDateTime).
    candidates = _get_all_clients_paginated(
        {"LastModifiedDate": modified_since, "IncludeInactive": "true"},
        max_pages=15,
    )

    signups = []
    cancellations = []
    seen_signup = set()       # (client_id, contract_id)
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

            start_date = (contract.get("StartDate") or "")[:10]
            if start_date and window_start <= start_date <= window_end:
                key = (client_id, contract_id)
                if key not in seen_signup:
                    seen_signup.add(key)
                    signups.append({
                        "client_id": client_id,
                        "contract_id": contract_id,
                        "name": client_name,
                        "membership": contract_name,
                        "date": start_date,
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
        "days_back": days_back,
        "window_start": window_start,
        "window_end": window_end,
        "signups": signups,
        "cancellations": cancellations,
    }
    _cache_set(cache_key, result)
    logger.info(
        f"Membership movement {days_back}d: "
        f"{len(signups)} signups, {len(cancellations)} cancellations "
        f"(scanned {len(candidates)} clients)"
    )
    return result
```

- [ ] **Step 2: Syntax-check the module**

Run: `python -c "import mindbody_helper; print('ok')"`
Expected: `ok` with no traceback.

- [ ] **Step 3: Confirm existing tests still pass**

Run: `python -m unittest discover tests -v`
Expected: all tests `ok`.

- [ ] **Step 4: Commit**

```bash
git add mindbody_helper.py
git commit -m "Add get_membership_movement data fetcher"
```

---

## Task 7: Wire the tool into `app.py`

**Files:**
- Modify: `app.py` — three edits

Line numbers below are approximate — find the anchor text shown before each edit and edit relative to it.

- [ ] **Step 1: Add the tool schema to `_MINDBODY_TOOLS`**

In `app.py`, locate the existing `get_new_members` entry (around line 275). Immediately after its closing brace (around line 284, before `get_arrears_report`), insert a new schema entry:

```python
    {
        "name": "get_membership_movement",
        "description": (
            "Signups AND cancellations of debiting memberships over a variable window. "
            "Use this for ANY cancellation/signup question spanning more than 30 days "
            "(e.g. 'cancellations and signups last 3 months', 'membership movement last 2 months'). "
            "ONLY debiting memberships count — casual passes, offers, and challenge memberships "
            "are excluded automatically. Pass split_by_month=true when the user asks for a monthly breakdown."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days_back": {
                    "type": "integer",
                    "description": "Days back. 30=last month, 90=last 3 months, 180=last 6 months. Default 90, max 365.",
                    "default": 90,
                },
                "split_by_month": {
                    "type": "boolean",
                    "description": "True when user asks for 'by month', 'broken down by month', 'month by month'. Default false.",
                    "default": False,
                },
            },
            "required": [],
        },
    },
```

- [ ] **Step 2: Add the routing branch in `handle_tool_call`**

In `app.py`, locate the existing `get_new_members` routing branch (around line 517–523). Immediately after its final `return`, insert:

```python
    elif tool_name == "get_membership_movement":
        from mindbody_helper import get_membership_movement, format_membership_movement
        days = min(int(tool_input.get("days_back", 90) or 90), 365)
        split = bool(tool_input.get("split_by_month", False))
        result = get_membership_movement(days_back=days)
        return format_membership_movement(result, days_back=days, split_by_month=split)
```

- [ ] **Step 3: Update `SYSTEM_PROMPT`**

In `app.py`, find the `SYSTEM_PROMPT` string (starts at line 68). Locate the line:

```python
    "MindBody is only for gym class schedules — use get_todays_classes or get_classes_history for that. "
```

Immediately after it, insert:

```python
    "IMPORTANT: For any cancellation or signup report covering more than 30 days "
    "(e.g. 'last 2 months', 'last 3 months', 'last 6 months'), you MUST use "
    "get_membership_movement. It only counts debiting memberships — casual passes, "
    "offers, and challenge memberships are excluded automatically. Pass "
    "split_by_month=true when the user asks for a monthly breakdown. "
```

- [ ] **Step 4: Syntax-check `app.py`**

Run: `python -c "import ast, sys; ast.parse(open('app.py').read()); print('ok')"`
Expected: `ok`.

Also run the unit tests: `python -m unittest discover tests -v`
Expected: all tests still pass.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "Wire get_membership_movement tool into WhatsApp bot"
```

---

## Task 8: Deploy and integration smoke test

**Files:** none (deploy + live tests)

- [ ] **Step 1: Deploy to Railway**

Run: `railway up --detach`
Expected: deploy completes, new revision live. Check Railway logs for any startup errors.

- [ ] **Step 2: Smoke test — flat mode, default window**

Send via WhatsApp to the bot: `"give me the membership movement for the last 3 months"`
Expected: reply starts with `*Membership Movement — Last 90 Days*`, shows `SIGNUPS: N`, `CANCELLATIONS: M`, grouped by membership with names, ends with `Net: ±K`. Verify first call takes <30s (slow-ack should fire — the existing `SLOW_KEYWORDS` contains `"cancellation"`, `"new signups"`, `"sign ups"`, `"report"` so this request already matches).

- [ ] **Step 3: Smoke test — monthly breakdown**

Send: `"and break it down by month"` (same conversation — history keeps context)
Expected: reply starts with `*Membership Movement — Last 90 Days (by month)*`, shows one section per calendar month with `── Month YYYY ──` headers, partial flag on edge months, totals footer.

- [ ] **Step 4: Smoke test — shorter window**

Send: `"signups and cancellations last month"`
Expected: reply says `Last 30 Days`. Verify Claude correctly extracts `days_back=30`.

- [ ] **Step 5: Spot-check correctness**

Pick one known recent signup and one known recent cancellation from MindBody (via the web UI or another tool). Confirm both appear in the correct bucket with the correct date and membership name. If they don't appear, check:
- Is the contract name in `TRACKED_MEMBERSHIPS`? (check `mindbody_helper.py` line ~291)
- Is the contract's `StartDate`/`TerminationDate` actually within the window?
- Does the client's `LastModifiedDateTime` fall within `days_back`? (candidates are filtered on this)

- [ ] **Step 6: Final commit / push**

No code changes expected at this step. If smoke tests revealed a bug, fix + re-commit + re-deploy before marking complete.

```bash
git log --oneline -5
```

---

## Notes on repo conventions honored

- **No mocking of MindBody** — follows the existing repo pattern (no mocked tests anywhere in this codebase). Live MindBody calls are smoke-tested in production.
- **Stdlib `unittest`, zero new deps** — `requirements.txt` is unchanged.
- **Lazy imports** in `handle_tool_call` — matches existing branches.
- **WhatsApp `*bold*` formatting** — matches existing formatters in `mindbody_helper.py`.
- **Cache key + TTL pattern** — matches `get_new_members` (`CACHE_TTL_CLASSES` = 1 hour).
- **`SLOW_KEYWORDS` not edited** — the existing entries (`"cancellation"`, `"report"`, `"sign ups"`, `"new signups"`) already match the phrases a user will use for this report, so the instant-ack fires automatically.
- **`TRACKED_MEMBERSHIPS` stays the single source of truth** — if a new debiting membership is launched, updating that list (mindbody_helper.py:291) is the only change needed; this report picks it up automatically.
