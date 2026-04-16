# Client Detail Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `get_client_detail` WhatsApp tool that returns a member's current membership, tenure, and class-attendance breakdown (all time / 30d / 90d / optional custom window).

**Architecture:** New `get_client_detail(client_name, days_back=None)` in `mindbody_helper.py` reuses existing `search_clients` for lookup, calls `client/activeclientmemberships` for memberships and `client/clientvisits` for attendance. `format_client_detail(result)` renders plain text. Wired into `app.py` via tool schema + `handle_tool_call` routing + `SYSTEM_PROMPT` guidance.

**Tech Stack:** Python 3, Flask, MindBody Public API v6, stdlib `unittest`.

**Spec:** `docs/superpowers/specs/2026-04-16-client-detail-design.md`

---

## File Structure

- **Modify** `mindbody_helper.py`
  - Add `get_client_detail(client_name, days_back=None)` — data fetcher
  - Add `format_client_detail(result)` — formatter (3 modes: found / not_found / multiple)
- **Modify** `app.py`
  - New tool schema entry in `_MINDBODY_TOOLS`
  - New routing branch in `handle_tool_call`
  - One new line in `SYSTEM_PROMPT`
- **Modify** `tests/test_membership_movement.py` → rename file conceptually; add `TestFormatClientDetail` class
  - Actually, to keep things simple: add tests to the existing test file since it already imports `mindbody_helper`

---

## Task 1: Tests for `format_client_detail` (TDD)

**Files:**
- Modify: `tests/test_membership_movement.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_membership_movement.py` (after `TestTruncation`, before `if __name__ == "__main__"`):

```python
class TestFormatClientDetail(unittest.TestCase):
    def test_found_basic(self):
        result = {
            "status": "found",
            "name": "Jane Smith",
            "member_since": "2024-03-14",
            "memberships": ["All Access 6 Month"],
            "classes_all_time": 247,
            "classes_30d": 18,
            "classes_90d": 52,
        }
        out = mindbody_helper.format_client_detail(result)
        self.assertIn("Jane Smith", out)
        self.assertIn("14 March 2024", out)
        self.assertIn("All Access 6 Month", out)
        self.assertIn("247 (all time)", out)
        self.assertIn("Last 30 days: 18", out)
        self.assertIn("Last 90 days: 52", out)

    def test_found_with_custom_window(self):
        result = {
            "status": "found",
            "name": "Jane Smith",
            "member_since": "2024-03-14",
            "memberships": ["All Access 6 Month"],
            "classes_all_time": 247,
            "classes_30d": 18,
            "classes_90d": 52,
            "classes_custom": 6,
            "classes_custom_label": 14,
        }
        out = mindbody_helper.format_client_detail(result)
        self.assertIn("Last 14 days: 6", out)
        self.assertIn("Last 30 days: 18", out)
        self.assertIn("Last 90 days: 52", out)

    def test_found_multiple_memberships(self):
        result = {
            "status": "found",
            "name": "Jane Smith",
            "member_since": "2024-03-14",
            "memberships": ["All Access 6 Month", "Student Membership"],
            "classes_all_time": 10,
            "classes_30d": 3,
            "classes_90d": 8,
        }
        out = mindbody_helper.format_client_detail(result)
        self.assertIn("Current memberships:", out)
        self.assertIn("All Access 6 Month", out)
        self.assertIn("Student Membership", out)

    def test_found_no_active_membership(self):
        result = {
            "status": "found",
            "name": "Jane Smith",
            "member_since": "2024-03-14",
            "memberships": [],
            "classes_all_time": 50,
            "classes_30d": 0,
            "classes_90d": 0,
        }
        out = mindbody_helper.format_client_detail(result)
        self.assertIn("No active membership", out)

    def test_found_zero_visits(self):
        result = {
            "status": "found",
            "name": "Jane Smith",
            "member_since": "2024-03-14",
            "memberships": ["All Access 6 Month"],
            "classes_all_time": 0,
            "classes_30d": 0,
            "classes_90d": 0,
        }
        out = mindbody_helper.format_client_detail(result)
        self.assertIn("0 (all time)", out)
        self.assertIn("Last 30 days: 0", out)

    def test_not_found(self):
        result = {
            "status": "not_found",
            "search_text": "Janee Smyth",
        }
        out = mindbody_helper.format_client_detail(result)
        self.assertIn("No client found", out)
        self.assertIn("Janee Smyth", out)

    def test_multiple_matches(self):
        result = {
            "status": "multiple",
            "matches": [
                {"name": "Jane Smith", "id": 123, "email": "jane@email.com"},
                {"name": "Jane Doe", "id": 456, "email": "jdoe@email.com"},
            ],
        }
        out = mindbody_helper.format_client_detail(result)
        self.assertIn("Found 2 matches", out)
        self.assertIn("Jane Smith", out)
        self.assertIn("Jane Doe", out)
        self.assertIn("Which one", out)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/samduncan/Desktop/claude-mcp && python3 -m unittest tests.test_membership_movement.TestFormatClientDetail -v`
Expected: All 7 tests fail with `AttributeError: module 'mindbody_helper' has no attribute 'format_client_detail'`.

- [ ] **Step 3: Commit test scaffolding**

```bash
git add tests/test_membership_movement.py
git commit -m "Add failing tests for format_client_detail (7 cases)"
```

---

## Task 2: Implement `format_client_detail` (TDD — green)

**Files:**
- Modify: `mindbody_helper.py` — add `format_client_detail()` in the Membership Movement Report section (after `_format_counts_block`, before `_truncate_to_whatsapp`)

- [ ] **Step 1: Implement the formatter**

Insert this function in `mindbody_helper.py` after `_format_counts_block` and before `_truncate_to_whatsapp`:

```python
# ── Client Detail ─────────────────────────────────────────────────────────────


def format_client_detail(result):
    """Format a client detail result dict for WhatsApp. Three modes:
    found (full profile), not_found, multiple (disambiguation)."""
    status = result.get("status")

    if status == "not_found":
        return (
            f'No client found for "{result["search_text"]}". '
            "Check the spelling or try a different name/email/phone."
        )

    if status == "multiple":
        matches = result["matches"]
        lines = [f'Found {len(matches)} matches for your search:']
        for i, m in enumerate(matches, 1):
            email = m.get("email") or "no email"
            lines.append(f"{i}. {m['name']} ({email})")
        lines.append("")
        lines.append("Which one did you mean?")
        return "\n".join(lines)

    # status == "found"
    date_str = _format_date(result["member_since"])
    lines = [result["name"]]
    lines.append(f"Member since: {date_str}")

    memberships = result.get("memberships") or []
    if len(memberships) == 0:
        lines.append("No active membership")
    elif len(memberships) == 1:
        lines.append(f"Current membership: {memberships[0]}")
    else:
        lines.append("Current memberships:")
        for m in memberships:
            lines.append(f"  {m}")

    lines.append("")
    lines.append(f"Classes attended: {result['classes_all_time']} (all time)")

    if "classes_custom" in result and "classes_custom_label" in result:
        lines.append(f"  Last {result['classes_custom_label']} days: {result['classes_custom']}")

    lines.append(f"  Last 30 days: {result['classes_30d']}")
    lines.append(f"  Last 90 days: {result['classes_90d']}")

    return "\n".join(lines)


def _format_date(iso_date):
    """Convert YYYY-MM-DD to '14 March 2024' for display."""
    if not iso_date or len(iso_date) < 10:
        return iso_date or "Unknown"
    try:
        from datetime import datetime as dt
        d = dt.strptime(iso_date[:10], "%Y-%m-%d")
        return d.strftime("%-d %B %Y")
    except (ValueError, TypeError):
        return iso_date
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /Users/samduncan/Desktop/claude-mcp && python3 -m unittest tests.test_membership_movement -v`
Expected: all tests pass (15 existing + 7 new = 22 total).

- [ ] **Step 3: Syntax check the module**

Run: `python3 -c "import mindbody_helper; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add mindbody_helper.py tests/test_membership_movement.py
git commit -m "Implement format_client_detail with 3-mode rendering"
```

---

## Task 3: Implement `get_client_detail` data fetcher

**Files:**
- Modify: `mindbody_helper.py` — add `get_client_detail()` after `format_client_detail`

This function calls live MindBody endpoints. Not unit-tested per repo convention (no mocks). Smoke-tested in Task 5.

- [ ] **Step 1: Add the data fetcher**

Insert after `_format_date`:

```python
def get_client_detail(client_name, days_back=None):
    """Look up a client by name and return their membership + attendance profile.

    Returns a result dict with status 'found', 'not_found', or 'multiple'.
    """
    clients = search_clients(client_name)

    if not clients:
        return {"status": "not_found", "search_text": client_name}

    if len(clients) > 1:
        return {
            "status": "multiple",
            "matches": [
                {"name": c["name"], "id": c["id"], "email": c["email"]}
                for c in clients[:10]
            ],
        }

    client = clients[0]
    client_id = client["id"]
    client_name_display = client["name"]

    # Member since — fetch full client record for CreationDate
    member_since = "Unknown"
    try:
        client_data = _api_get("client/clients", {"ClientIds": client_id})
        full_clients = client_data.get("Clients") or []
        if full_clients:
            member_since = (full_clients[0].get("CreationDate") or "")[:10] or "Unknown"
    except Exception as e:
        logger.warning(f"Client lookup failed for {client_id}: {e}")

    # Active memberships
    memberships = []
    try:
        mem_data = _api_get("client/activeclientmemberships", {"ClientIds": client_id})
        for cm in mem_data.get("ClientMemberships") or []:
            for m in cm.get("Memberships") or []:
                name = m.get("Name")
                if name:
                    memberships.append(name)
    except Exception as e:
        logger.warning(f"Membership lookup failed for {client_id}: {e}")

    # Visit history — fetch all visits from a generous start date
    now = _now()
    today_iso = now.strftime("%Y-%m-%dT23:59:59")
    all_visits = []
    try:
        visit_data = _paginated_get(
            "client/clientvisits", "Visits",
            {"ClientId": client_id, "StartDate": "2010-01-01T00:00:00", "EndDate": today_iso},
            max_pages=10,
        )
        all_visits = visit_data if isinstance(visit_data, list) else []
    except Exception as e:
        logger.warning(f"Visit lookup failed for {client_id}: {e}")

    # Count attended visits (SignedIn == True) per window
    cutoff_30 = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    cutoff_90 = (now - timedelta(days=90)).strftime("%Y-%m-%d")
    cutoff_custom = (now - timedelta(days=days_back)).strftime("%Y-%m-%d") if days_back and days_back > 0 else None

    classes_all = 0
    classes_30 = 0
    classes_90 = 0
    classes_custom = 0

    for v in all_visits:
        if not v.get("SignedIn", False):
            continue
        classes_all += 1
        visit_date = (v.get("StartDateTime") or "")[:10]
        if visit_date >= cutoff_90:
            classes_90 += 1
        if visit_date >= cutoff_30:
            classes_30 += 1
        if cutoff_custom and visit_date >= cutoff_custom:
            classes_custom += 1

    result = {
        "status": "found",
        "name": client_name_display,
        "member_since": member_since,
        "memberships": memberships,
        "classes_all_time": classes_all,
        "classes_30d": classes_30,
        "classes_90d": classes_90,
    }

    if days_back and days_back > 0:
        result["classes_custom"] = classes_custom
        result["classes_custom_label"] = days_back

    logger.info(
        f"Client detail for {client_name_display}: "
        f"{len(memberships)} memberships, {classes_all} total visits "
        f"({classes_30} in 30d, {classes_90} in 90d)"
    )
    return result
```

- [ ] **Step 2: Syntax check the module**

Run: `python3 -c "import mindbody_helper; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Confirm existing tests still pass**

Run: `cd /Users/samduncan/Desktop/claude-mcp && python3 -m unittest discover tests -v`
Expected: all 22 tests pass.

- [ ] **Step 4: Commit**

```bash
git add mindbody_helper.py
git commit -m "Add get_client_detail data fetcher"
```

---

## Task 4: Wire the tool into `app.py`

**Files:**
- Modify: `app.py` — three edits

- [ ] **Step 1: Add the tool schema to `_MINDBODY_TOOLS`**

In `app.py`, find the existing `search_clients` tool entry (it has `"name": "search_clients"`). Immediately after it, insert:

```python
    {
        "name": "get_client_detail",
        "description": (
            "Detailed profile for a specific client: current membership, how long they have "
            "been a member, and class attendance breakdown (all time, 30 days, 90 days). "
            "Use when the user asks about a specific member's details, classes done, membership, "
            "or how long they've been coming. Pass the client's name as client_name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "client_name": {
                    "type": "string",
                    "description": "Client name (or partial name, email, phone) to search for",
                },
                "days_back": {
                    "type": "integer",
                    "description": "Optional custom window in days (e.g. 7 for last week, 14 for last 2 weeks). Omit for default 30d/90d/all-time only.",
                },
            },
            "required": ["client_name"],
        },
    },
```

- [ ] **Step 2: Add the routing branch in `handle_tool_call`**

In `app.py`, find the existing `search_clients` routing branch (`elif tool_name == "search_clients":`). Immediately after that block's `return`, insert:

```python
    elif tool_name == "get_client_detail":
        from mindbody_helper import get_client_detail, format_client_detail
        name = (tool_input.get("client_name") or "")[:100]
        days = tool_input.get("days_back")
        if days is not None:
            days = max(1, min(int(days or 0), 365))
        result = get_client_detail(client_name=name, days_back=days)
        return format_client_detail(result)
```

- [ ] **Step 3: Update `SYSTEM_PROMPT`**

In `app.py`, find the `SYSTEM_PROMPT` string. Locate the line that starts with:

```python
    "IMPORTANT: When the user asks for a 'membership report'"
```

Immediately BEFORE that line, insert:

```python
    "When the user asks about a specific member's details, membership, how long they've been "
    "a member, or how many classes they've done, use get_client_detail with their name. "
    "Do NOT use search_clients for this — search_clients only finds a member, it doesn't "
    "return membership or attendance data. "
```

- [ ] **Step 4: Syntax check**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('ok')"`
Expected: `ok`.

Run: `python3 -m unittest discover tests -v`
Expected: all 22 tests pass.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "Wire get_client_detail tool into WhatsApp bot"
```

---

## Task 5: Deploy and integration smoke test

**Files:** none (deploy + live tests)

- [ ] **Step 1: Deploy to Railway**

Run: `railway up --detach`
Expected: deploy completes, new revision live.

- [ ] **Step 2: Smoke test — basic client lookup**

Send via WhatsApp: `"how many classes has [known member name] done?"`
Expected: reply with client name, member since date, current membership, and three class-count windows (all time / 30d / 90d).

- [ ] **Step 3: Smoke test — custom window**

Send: `"how many classes has [same member] done in the last 2 weeks?"`
Expected: reply includes `Last 14 days: N` line alongside the default windows.

- [ ] **Step 4: Smoke test — ambiguous name**

Send: `"give me details for [common first name only]"`
Expected: disambiguation list with numbered matches and "Which one did you mean?"

- [ ] **Step 5: Smoke test — non-existent client**

Send: `"how many classes has Zzzzznotreal Fakename done?"`
Expected: `No client found for "Zzzzznotreal Fakename".`

- [ ] **Step 6: Cross-check correctness**

Pick the client from step 2. Manually check their class count against the MindBody web UI. Verify the all-time count is close (may differ by a few if MindBody's web UI counts differently). Verify member-since date matches.

---

## Notes on repo conventions honored

- **No mocking of MindBody** — data fetcher is smoke-tested live.
- **Stdlib `unittest`** — tests added to existing `tests/test_membership_movement.py`.
- **Lazy imports** in `handle_tool_call` — matches existing branches.
- **Plain text formatting** — no `*` bold, no decoration (matches the membership report style after user feedback).
- **`python3`** on this machine, not `python`.
- **Cache not used for client detail** — single-client lookups are fast (~1-3 seconds) and the user is likely asking about different clients each time. No benefit from caching stale data.
- **`search_clients` reuse** — the existing search function already handles the MindBody search API; `get_client_detail` calls it directly rather than duplicating the logic.
