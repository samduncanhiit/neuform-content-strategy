# MindBody-Only WhatsApp Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce this repo to the HIIT Station WhatsApp bot answering MindBody questions only, after archiving everything that is removed under a git tag.

**Architecture:** No restructuring. Finish the in-flight branch, commit all untracked files as an archive and tag it, then on a new branch strip every non-MindBody tool, route, scheduler, config and file from `app.py` and the repo. The revenue password moves from source into the `REVENUE_PASSWORD` env var and fails closed when unset. A new test file pins the MindBody-only surface (tool list, handlers, routes, prompt, files, dependencies).

**Tech Stack:** Python 3 (local 3.9, Railway 3.11), Flask, Twilio, Anthropic SDK, `unittest` run via `python3 -m pytest`.

**Spec:** `docs/superpowers/specs/2026-09-29-mindbody-only-design.md`

## Global Constraints

- Everything removed must first be committed and tagged `pre-mindbody-only`. Never delete an untracked file before the archive commit exists.
- The strip happens on branch `chore/mindbody-only`, branched from the archive commit.
- `mindbody_helper.py` is not modified after Task 1.
- The 14 kept tools, exactly: `get_todays_classes`, `get_daily_briefing`, `search_clients`, `get_client_detail`, `get_member_stats`, `get_payment_failures`, `get_classes_history`, `get_revenue`, `get_new_members`, `get_membership_movement`, `get_arrears_report`, `get_weekly_summary`, `run_class_report`, `get_noshow_report`.
- Kept routes, exactly: `/webhook`, `/health`, `/`.
- Revenue password env var name: `REVENUE_PASSWORD`. Unset, empty or whitespace-only means fail closed (Erin gets no revenue, no password is offered).
- No `git push`, no `railway up`, no Railway dashboard changes. The user deploys.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- The gitignored local files `docs/HIIT-Operations-Manual.docx`, `docs/HIIT-Station-Operations-Manual.docx` and `hiit-bot-*.json` are left on disk untouched. They are not in git, so the archive tag cannot preserve them; deleting them would be permanent. (Deviation from the spec's "remove local .docx" — flag to the user in the final report.)

## Review Focus

1. **Stale history names a removed tool** — history lives in memory for 15 minutes, so a model could still emit e.g. `read_inbox` right after a hot reload. Expected: `handle_tool_call` returns `Unknown tool: read_inbox`, no crash, no import of a deleted module. Pinned in Task 3 (`test_removed_tools_are_unknown`).
2. **`REVENUE_PASSWORD` set to whitespace** (a stray space pasted into Railway). Expected: treated as unset, fail closed, never "Correct password: ' '". Pinned in Task 4 (`test_whitespace_password_fails_closed`).
3. **Railway build from a clean checkout** — the deployed image only has what `requirements.txt` lists. Expected: `import app` works in a fresh venv built from `requirements.txt` alone. Pinned in Task 6 Step 4.
4. **"connect gmail" sent after the change** — the command handler is gone. Expected: it goes to Claude like any message and gets the one-line off-topic refusal; nothing imports `gmail_helper`. Pinned in Task 3 (`test_no_removed_module_referenced_in_app`) and Task 5 (`test_removed_modules_are_gone`).
5. **Erin asks for the weekly summary** — `get_weekly_summary` includes revenue, so Erin can see revenue without the password. This is pre-existing and out of scope (spec: no behaviour change to MindBody tools); no test. Report it to the user in the final summary as a follow-up.

---

### Task 1: Finish the in-flight cancellation dedup fix

**Files:**
- Modify (already edited, uncommitted): `mindbody_helper.py:1289-1296`
- Test (already edited, uncommitted): `tests/test_membership_movement.py:667-708`

**Interfaces:**
- Consumes: nothing.
- Produces: a clean `mindbody_helper.py` on `feat/membership-movement` that later tasks never touch.

- [ ] **Step 1: Confirm branch and the exact pending diff**

Run: `git branch --show-current && git diff --stat`
Expected: branch `feat/membership-movement`; diff lists `mindbody_helper.py`, `tests/test_membership_movement.py`, and deleted `index.html`, `neuform-content-calendar.html`. Nothing else modified.

- [ ] **Step 2: Run the dedup test on its own**

Run: `python3 -m pytest tests/test_membership_movement.py -k duplicate_contract_rows -v`
Expected: PASS (1 test).

- [ ] **Step 3: Run the full suite**

Run: `python3 -m pytest -q`
Expected: `123 passed`.

- [ ] **Step 4: Commit only the two MindBody files**

```bash
git add mindbody_helper.py tests/test_membership_movement.py
git commit -m "fix(mindbody): count duplicate contract rows as one cancellation

Auto-renewing memberships get the same TerminationDate on the expiring
term and the generated renewal row. Dedup cancellations by
client + membership + date instead of contract id.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Run: `git status --porcelain`
Expected: only the two ` D` root HTML files and the `??` untracked entries remain.

---

### Task 2: Archive everything and tag it

**Files:**
- Commit as-is (currently untracked): `.railwayignore`, `CLAUDE.md`, `content-calendar/`, `docs/CLAUDE-CODE-PROMPT-OUTLOOK-SORT.md`, `docs/CLAUDE-CODE-PROMPTS.md`, `docs/HIIT-Station-Operations-Manual.md`, `docs/create_lists.py`, `docs/generate_manual.py`, `docs/list_contracts.py`, `docs/list_trello_boards.py`, `docs/mindbody_briefing.py`, `gcal_helper.py`, `gmail_helper.py`, `lead_automation.py`, `outlook_helper.py`, `requirements.txt`, `runtime.txt`, `docs/superpowers/plans/2026-09-29-mindbody-only.md` (if not already committed)
- Commit deletion: `index.html`, `neuform-content-calendar.html`

**Interfaces:**
- Consumes: Task 1's clean branch.
- Produces: tag `pre-mindbody-only` pointing at a commit where every file that exists today is tracked.

- [ ] **Step 1: Confirm secrets stay ignored**

Run: `git status --porcelain --ignored | grep -E 'hiit-bot|\.env|\.docx'`
Expected: every line starts with `!!` (ignored). If any line starts with `??`, STOP and ask the user.

- [ ] **Step 2: Stage everything**

```bash
git add -A
```

- [ ] **Step 3: Scan what is staged for credentials**

Run: `git diff --cached --name-only`
Expected: exactly the files listed under **Files** above (plus the content-calendar contents). No `.json` other than `content-calendar/drive_folder_map.json` and `content-calendar/viral-fitness-videos.json`.

Run: `git diff --cached | grep -nEi '(sk-ant-|AC[0-9a-f]{32}|"private_key"|client_secret"\s*:|api_key\s*=\s*["'"'"'][A-Za-z0-9])' || echo CLEAN`
Expected: `CLEAN`. If anything matches, STOP, unstage (`git reset`), and ask the user.

- [ ] **Step 4: Commit and tag**

```bash
git commit -m "chore: archive all local files before MindBody-only cleanup

Commits every previously untracked file (helpers the bot imports,
content calendar, docs, requirements) so the MindBody-only strip is
recoverable from tag pre-mindbody-only.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git tag -a pre-mindbody-only -m "Snapshot before stripping the repo to the MindBody-only WhatsApp bot"
```

Run: `git status --porcelain && git tag -l pre-mindbody-only`
Expected: status prints nothing; tag prints `pre-mindbody-only`.

- [ ] **Step 5: Confirm the suite still passes at the tag**

Run: `python3 -m pytest -q`
Expected: `123 passed`.

- [ ] **Step 6: Create the working branch**

```bash
git checkout -b chore/mindbody-only
```

---

### Task 3: Strip `app.py` to MindBody tools, routes and prompt

**Files:**
- Create: `tests/test_mindbody_only.py`
- Modify: `app.py` (config block lines ~27-56, `SYSTEM_PROMPT` ~58-114, tool definitions ~260-663, `handle_tool_call` ~665-1002, `_build_system_prompt` ~1005-1048, `get_claude_response` ~1065-1096, `SLOW_KEYWORDS` ~1208-1220, `process_message_async` ~1239-1244, routes ~1308-1544, scheduler ~1557-1626)
- Delete: `tests/test_trello_write.py`, `tests/test_jotform_helper.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `app._MINDBODY_TOOLS` — list of 14 tool dicts, the only tools sent to Claude.
  - `app.handle_tool_call(tool_name, tool_input) -> str` — two parameters only.
  - `app._build_system_prompt(user_name, raw_number) -> str` — unchanged signature; Task 4 edits its Erin block.
  - `tests/test_mindbody_only.py` with module constants `EXPECTED_TOOLS`, `REMOVED_TOOLS`, `ERIN`, `SAM` — Tasks 4 and 5 add test classes to this file.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mindbody_only.py`:

```python
"""Pins the bot to MindBody-only scope: tools, handlers, routes, prompt."""
import contextlib
import inspect
import os
import unittest
from unittest.mock import patch

import app
import mindbody_helper

EXPECTED_TOOLS = {
    "get_todays_classes", "get_daily_briefing", "search_clients",
    "get_client_detail", "get_member_stats", "get_payment_failures",
    "get_classes_history", "get_revenue", "get_new_members",
    "get_membership_movement", "get_arrears_report", "get_weekly_summary",
    "run_class_report", "get_noshow_report",
}

REMOVED_TOOLS = [
    "get_trello_tasks", "add_trello_card", "edit_trello_card",
    "move_trello_card", "remove_trello_card", "get_jotform_submissions",
    "get_calendar_events", "create_calendar_event", "read_inbox",
    "draft_email", "read_gmail", "read_gmail_drafts",
]

REMOVED_MODULES = [
    "gcal_helper", "gmail_helper", "outlook_helper",
    "trello_helper", "jotform_helper", "lead_automation",
]

ERIN = "+61421188443"
SAM = "+61420233508"

APP_SOURCE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@contextlib.contextmanager
def stubbed_mindbody():
    """Replace every public mindbody_helper function with a stub returning '<name>'."""
    with contextlib.ExitStack() as stack:
        for name, obj in list(vars(mindbody_helper).items()):
            if inspect.isfunction(obj) and not name.startswith("_"):
                stack.enter_context(
                    patch.object(mindbody_helper, name, return_value=f"<{name}>")
                )
        yield


class TestToolSurface(unittest.TestCase):
    def test_tool_list_is_exactly_mindbody(self):
        names = {t["name"] for t in app._MINDBODY_TOOLS}
        self.assertEqual(names, EXPECTED_TOOLS)

    def test_every_tool_has_a_handler(self):
        with stubbed_mindbody():
            for name in sorted(EXPECTED_TOOLS):
                with self.subTest(tool=name):
                    result = app.handle_tool_call(name, {})
                    self.assertFalse(result.startswith("Unknown tool"), result)

    def test_removed_tools_are_unknown(self):
        for name in REMOVED_TOOLS:
            with self.subTest(tool=name):
                self.assertEqual(app.handle_tool_call(name, {}), f"Unknown tool: {name}")

    def test_handle_tool_call_takes_only_name_and_input(self):
        params = list(inspect.signature(app.handle_tool_call).parameters)
        self.assertEqual(params, ["tool_name", "tool_input"])

    def test_daily_briefing_is_classes_only(self):
        with stubbed_mindbody():
            self.assertEqual(app.handle_tool_call("get_daily_briefing", {}), "<format_briefing>")

    def test_daily_briefing_description_has_no_calendar_or_inbox(self):
        tool = next(t for t in app._MINDBODY_TOOLS if t["name"] == "get_daily_briefing")
        self.assertNotIn("calendar", tool["description"].lower())
        self.assertNotIn("inbox", tool["description"].lower())


class TestAppSurface(unittest.TestCase):
    def test_only_mindbody_routes_remain(self):
        rules = {r.rule for r in app.app.url_map.iter_rules() if r.endpoint != "static"}
        self.assertEqual(rules, {"/webhook", "/health", "/"})

    def test_lead_scheduler_removed(self):
        self.assertFalse(hasattr(app, "start_scheduler"))
        self.assertFalse(hasattr(app, "_run_daily_leads"))

    def test_per_user_integration_config_removed(self):
        for attr in ("USER_EMAILS", "USER_GMAIL", "USER_CALENDAR", "USER_TRELLO",
                     "_get_tools_for_user", "ALL_TOOLS"):
            with self.subTest(attr=attr):
                self.assertFalse(hasattr(app, attr))

    def test_slow_keywords_have_no_removed_features(self):
        for kw in ("trello", "hiit challenge", "tasks", "submission",
                   "submissions", "jotform", "form submissions"):
            with self.subTest(kw=kw):
                self.assertNotIn(kw, app.SLOW_KEYWORDS)

    def test_no_removed_module_referenced_in_app(self):
        with open(APP_SOURCE_PATH) as f:
            source = f.read()
        for mod in REMOVED_MODULES:
            with self.subTest(module=mod):
                self.assertNotIn(mod, source)


class TestSystemPrompt(unittest.TestCase):
    def test_prompt_declares_mindbody_only_scope(self):
        prompt = app._build_system_prompt("Sam", SAM)
        self.assertIn("this bot only handles MindBody questions", prompt)

    def test_prompt_has_no_removed_tool_guidance(self):
        prompt = app._build_system_prompt("Sam", SAM)
        for phrase in ("get_calendar_events", "read_inbox", "read_gmail",
                       "add_trello_card", "move_trello_card", "remove_trello_card",
                       "get_jotform_submissions", "create a draft",
                       "Prefix event name", "two email accounts"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, prompt)

    def test_prompt_keeps_mindbody_routing_rules(self):
        prompt = app._build_system_prompt("Sam", SAM)
        for phrase in ("CRITICAL RULE — verbatim tool output", "get_membership_movement",
                       "get_client_detail", "get_noshow_report", "FORMATTING:"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, prompt)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Delete the tests for removed features**

```bash
git rm tests/test_trello_write.py tests/test_jotform_helper.py
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `python3 -m pytest tests/test_mindbody_only.py -q`
Expected: FAIL/ERROR on most tests (e.g. `test_tool_list_is_exactly_mindbody` sees `get_trello_tasks`; `test_handle_tool_call_takes_only_name_and_input` sees `user_email`, `raw_number`; `test_only_mindbody_routes_remain` sees `/cron/leads`). `test_removed_tools_are_unknown` may ERROR on network/credential exceptions rather than FAIL — that is fine.

- [ ] **Step 4: Remove the `requests` import and per-user integration config**

In `app.py`, delete the line `import requests` (it is only used by the OAuth callback being removed; `mindbody_helper.py` imports its own).

Delete these four blocks entirely (from `USER_EMAILS = {` through the closing `}` of `USER_TRELLO`), keeping `USER_NAMES` above them:

```python
USER_EMAILS = { ... }
# Gmail accounts (separate from Outlook)
USER_GMAIL = { ... }
# Google Calendar IDs per user
USER_CALENDAR = { ... }
# Per-user Trello config. Only users in this dict get Trello write access.
USER_TRELLO = { ... }
```

- [ ] **Step 5: Replace `SYSTEM_PROMPT`**

Replace the whole `SYSTEM_PROMPT = ( ... )` assignment with:

```python
SYSTEM_PROMPT = (
    "You are an AI assistant for HIIT Station Capalaba. You answer questions about the gym's "
    "MindBody data: classes, bookings, members, payments, revenue, and reports. "
    "Keep responses brief and suitable for WhatsApp messaging.\n\n"
    "SCOPE: If the user asks for anything outside MindBody (email, calendar, Trello, forms, "
    "or general tasks), do not call any tool and reply with exactly one line: "
    "'Sorry, this bot only handles MindBody questions.'\n\n"
    "CRITICAL RULE — verbatim tool output: When the get_membership_movement tool returns a "
    "result, your ENTIRE reply to the user MUST be exactly that tool output, character-for-"
    "character. Do NOT add a greeting, introduction, 'Hey Sam', heading, summary, closing "
    "note, emoji, or any commentary before or after. Do NOT shorten, abbreviate, reorder, "
    "rewrite headings, change bullet characters, or drop sections. Do NOT insert '(truncated)' "
    "or any other marker. Copy the tool output into your reply exactly as received and stop.\n\n"
    "You have access to MindBody tools (classes, members, client detail, revenue, payments, "
    "new member signups, membership movement, arrears report, weekly summary, class reports, "
    "no-show reports). "
    "For gym class schedules use get_todays_classes or get_classes_history. "
    "When the user asks about a specific member's details, membership, how long they've been "
    "a member, or how many classes they've done, use get_client_detail with their name. "
    "Do NOT use search_clients for this — search_clients only finds a member, it doesn't "
    "return membership or attendance data. "
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
    "IMPORTANT: When the user asks about no-shows, who didn't show up, who didn't sign in, "
    "or who didn't attend a class, you MUST use the get_noshow_report tool. "
    "Do NOT use get_todays_classes or get_classes_history for this — those only show booking counts. "
    "get_noshow_report checks the actual sign-in roster and returns individual client names. "
    "FORMATTING: All responses must be plain text suitable for copy-pasting into other chats. "
    "Never use markdown tables, horizontal lines (---), pipes (|), or special formatting. "
    "Use simple lists with numbers or bullet points. Use *bold* for headings only. "
    "Keep it clean and easy to copy-paste."
)
```

- [ ] **Step 6: Trim the tool definitions**

1. Replace the comment line `# Organised by category so we can send only the tools each user needs.` with nothing (delete it).
2. In `_MINDBODY_TOOLS`, change the `get_daily_briefing` description to:

```python
        "description": "Daily briefing: today's classes and bookings",
```

3. In `_MINDBODY_TOOLS`, delete the whole `get_trello_tasks` entry:

```python
    {
        "name": "get_trello_tasks",
        "description": "HIIT Challenge Trello cards due today or overdue",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
```

4. Delete everything from `_CALENDAR_TOOLS = [` down to and including the end of `def _get_user_gmail(...)` (its final `return user_email`). That removes `_CALENDAR_TOOLS`, `_OUTLOOK_TOOLS`, `_GMAIL_TOOLS`, `_GMAIL_USERS`, `_TRELLO_WRITE_TOOLS`, `_JOTFORM_TOOLS`, `_get_tools_for_user`, `ALL_TOOLS`, `_get_user_calendar_ids`, `_get_user_gmail`.

- [ ] **Step 7: Trim `handle_tool_call`**

Replace the signature line and the `get_daily_briefing` branch (from `def handle_tool_call(` through the `return "\n\n".join(parts)` that ends the briefing branch) with:

```python
def handle_tool_call(tool_name, tool_input):
    """Execute a tool call and return the result."""
    if tool_name == "get_todays_classes":
        from mindbody_helper import get_todays_schedule, format_schedule
        classes = get_todays_schedule()
        return format_schedule(classes)

    elif tool_name == "get_daily_briefing":
        from mindbody_helper import get_daily_briefing, format_briefing
        return format_briefing(get_daily_briefing())
```

Then delete every branch from `elif tool_name == "get_trello_tasks":` down to (not including) the final `return f"Unknown tool: {tool_name}"`. That removes the Trello, JotForm, Google Calendar, Outlook and Gmail branches. The `get_noshow_report` branch is now the last `elif`.

- [ ] **Step 8: Trim `_build_system_prompt`**

Delete the Chonnie block:

```python
    # Chonnie has both Outlook and Gmail
    if raw_number == "+61481123186":
        parts.append( ... )
```

Delete the calendar-event block at the end:

```python
    if user_name:
        parts.append(
            "When creating calendar events: create immediately, ..."
            ...
        )
```

Leave the Erin revenue block unchanged (Task 4 owns it).

- [ ] **Step 9: Trim `get_claude_response`**

Replace:

```python
    user_name = USER_NAMES.get(raw_number)
    user_email = USER_EMAILS.get(raw_number, "sam@hiitaustralia.com.au")
    system = _build_system_prompt(user_name, raw_number)
    tools = _get_tools_for_user(user_email, raw_number=raw_number)
```

with:

```python
    user_name = USER_NAMES.get(raw_number)
    system = _build_system_prompt(user_name, raw_number)
    tools = _MINDBODY_TOOLS
```

Replace:

```python
                    result = handle_tool_call(block.name, block.input, user_email=user_email, raw_number=raw_number)
```

with:

```python
                    result = handle_tool_call(block.name, block.input)
```

- [ ] **Step 10: Trim `SLOW_KEYWORDS` and `process_message_async`**

In `SLOW_KEYWORDS`, delete these two lines:

```python
    "trello", "hiit challenge", "tasks",
    "submission", "submissions", "jotform", "form submissions",
```

In `process_message_async`, delete the Gmail connect block:

```python
    # Handle Gmail connect command
    if incoming_msg.lower().strip() in ("connect gmail", "setup gmail", "link gmail"):
        from gmail_helper import get_auth_url
        auth_url = get_auth_url()
        send_whatsapp_reply(sender, f"Click this link to connect your Gmail:\n\n{auth_url}")
        return
```

- [ ] **Step 11: Remove the non-MindBody routes and the scheduler**

1. Delete everything from `@app.route("/oauth/callback", methods=["GET"])` down to (not including) `@app.route("/health", methods=["GET"])`. That removes `oauth_callback`, `CRON_SECRET`, `cron_leads`, `cron_leads_debug`, the `# ── Neuform Content Upload API` section, `drive_upload`, `drive_folders`.
2. Delete everything from `# ── Background scheduler: daily lead automation at 5am AEST` down to (not including) `# ── Entry point`. That removes `_scheduler_started`, `_run_daily_leads`, `start_scheduler`, and the `if os.environ.get("PORT"): start_scheduler()` block.
3. In the `if __name__ == "__main__":` block, delete the line `    start_scheduler()`. The block becomes:

```python
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
```

- [ ] **Step 12: Run the new tests**

Run: `python3 -m pytest tests/test_mindbody_only.py -q`
Expected: all PASS.

- [ ] **Step 13: Run the full suite**

Run: `python3 -m pytest -q`
Expected: all pass (membership-movement tests + `test_mindbody_only.py`), 0 failures.

- [ ] **Step 14: Commit**

```bash
git add app.py tests/test_mindbody_only.py
git commit -m "refactor(app): strip bot to MindBody-only tools, routes and prompt

Removes Trello, JotForm, Google Calendar, Outlook and Gmail tools, the
lead-automation scheduler and cron routes, the Gmail OAuth callback and
the Neuform Drive upload API. Daily briefing is classes-only. The system
prompt refuses non-MindBody requests in one line.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Read Erin's revenue password from `REVENUE_PASSWORD`

**Files:**
- Modify: `app.py` (`_build_system_prompt`, the `# Erin: password-protect revenue data` block)
- Test: `tests/test_mindbody_only.py` (append class)

**Interfaces:**
- Consumes: `app._build_system_prompt(user_name, raw_number) -> str`, constants `ERIN`, `SAM`, `APP_SOURCE_PATH` from Task 3's test file.
- Produces: env var contract `REVENUE_PASSWORD` (read at call time, stripped; empty means unset).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mindbody_only.py`, above the `if __name__ == "__main__":` line:

```python
class TestRevenuePassword(unittest.TestCase):
    def _prompt(self, raw_number, env):
        with patch.dict(os.environ, env, clear=False):
            if "REVENUE_PASSWORD" not in env:
                os.environ.pop("REVENUE_PASSWORD", None)
            return app._build_system_prompt("Erin" if raw_number == ERIN else "Sam", raw_number)

    def test_password_comes_from_env(self):
        prompt = self._prompt(ERIN, {"REVENUE_PASSWORD": "hunter2"})
        self.assertIn("Correct password: 'hunter2'", prompt)

    def test_password_is_stripped(self):
        prompt = self._prompt(ERIN, {"REVENUE_PASSWORD": "  hunter2\n"})
        self.assertIn("Correct password: 'hunter2'", prompt)

    def test_unset_password_fails_closed(self):
        prompt = self._prompt(ERIN, {})
        self.assertNotIn("Correct password", prompt)
        self.assertIn("Never call get_revenue for this user", prompt)

    def test_whitespace_password_fails_closed(self):
        prompt = self._prompt(ERIN, {"REVENUE_PASSWORD": "   "})
        self.assertNotIn("Correct password", prompt)
        self.assertIn("Never call get_revenue for this user", prompt)

    def test_other_users_never_see_password_text(self):
        prompt = self._prompt(SAM, {"REVENUE_PASSWORD": "hunter2"})
        self.assertNotIn("hunter2", prompt)
        self.assertNotIn("get_revenue for this user", prompt)

    def test_old_password_not_in_source(self):
        with open(APP_SOURCE_PATH) as f:
            self.assertNotIn("samistheman", f.read())
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_mindbody_only.py -k RevenuePassword -q`
Expected: FAIL on `test_password_comes_from_env`, `test_password_is_stripped`, `test_unset_password_fails_closed`, `test_whitespace_password_fails_closed`, `test_old_password_not_in_source`. `test_other_users_never_see_password_text` passes already.

- [ ] **Step 3: Implement**

In `_build_system_prompt`, replace:

```python
    # Erin: password-protect revenue data
    if raw_number == "+61421188443":
        parts.append(
            "IMPORTANT: This user does NOT have access to revenue data. "
            "If they ask about revenue, income, money, debits, or financial reports, "
            "ask for a password first. Correct password: 'samistheman'. "
            "Only call get_revenue if they give the exact password."
        )
```

with:

```python
    # Erin: revenue is password-protected. Fail closed if REVENUE_PASSWORD is unset.
    if raw_number == "+61421188443":
        revenue_password = os.environ.get("REVENUE_PASSWORD", "").strip()
        if revenue_password:
            parts.append(
                "IMPORTANT: This user does NOT have access to revenue data. "
                "If they ask about revenue, income, money, debits, or financial reports, "
                f"ask for a password first. Correct password: '{revenue_password}'. "
                "Only call get_revenue if they give the exact password."
            )
        else:
            parts.append(
                "IMPORTANT: This user does NOT have access to revenue data. "
                "If they ask about revenue, income, money, debits, or financial reports, "
                "tell them revenue isn't available to them. "
                "Never call get_revenue for this user."
            )
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest -q`
Expected: all pass, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_mindbody_only.py
git commit -m "fix(app): read revenue password from REVENUE_PASSWORD env var

Removes the hardcoded password from source. When the env var is unset
or blank, Erin is told revenue is unavailable and get_revenue is never
offered.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Delete non-MindBody files, docs and dependencies

**Files:**
- Delete: `gcal_helper.py`, `gmail_helper.py`, `outlook_helper.py`, `trello_helper.py`, `jotform_helper.py`, `lead_automation.py`, `.railwayignore`, `content-calendar/` (whole directory), `docs/CLAUDE-CODE-PROMPT-OUTLOOK-SORT.md`, `docs/CLAUDE-CODE-PROMPTS.md`, `docs/HIIT-Station-Operations-Manual.md`, `docs/create_lists.py`, `docs/generate_manual.py`, `docs/list_trello_boards.py`, `docs/mindbody_briefing.py`
- Keep: `docs/list_contracts.py`, `docs/superpowers/`
- Modify: `requirements.txt`
- Test: `tests/test_mindbody_only.py` (append class)

**Interfaces:**
- Consumes: `REMOVED_MODULES` from Task 3's test file.
- Produces: a repo whose only Python modules are `app.py`, `mindbody_helper.py`, `docs/list_contracts.py`, and `tests/`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mindbody_only.py`, above the `if __name__ == "__main__":` line:

```python
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestRepoContents(unittest.TestCase):
    def test_removed_modules_are_gone(self):
        for mod in REMOVED_MODULES:
            with self.subTest(module=mod):
                self.assertFalse(os.path.exists(os.path.join(REPO_ROOT, f"{mod}.py")))

    def test_content_calendar_is_gone(self):
        self.assertFalse(os.path.exists(os.path.join(REPO_ROOT, "content-calendar")))

    def test_mindbody_maintenance_script_kept(self):
        self.assertTrue(os.path.exists(os.path.join(REPO_ROOT, "docs", "list_contracts.py")))

    def test_requirements_are_mindbody_only(self):
        with open(os.path.join(REPO_ROOT, "requirements.txt")) as f:
            packages = {line.split("==")[0].strip().lower()
                        for line in f if line.strip() and not line.startswith("#")}
        self.assertEqual(packages, {"requests", "flask", "twilio", "anthropic", "gunicorn"})
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_mindbody_only.py -k RepoContents -q`
Expected: FAIL on `test_removed_modules_are_gone`, `test_content_calendar_is_gone`, `test_requirements_are_mindbody_only`; `test_mindbody_maintenance_script_kept` passes.

- [ ] **Step 3: Delete the files**

```bash
git rm -q gcal_helper.py gmail_helper.py outlook_helper.py trello_helper.py \
  jotform_helper.py lead_automation.py .railwayignore
git rm -rq content-calendar
git rm -q docs/CLAUDE-CODE-PROMPT-OUTLOOK-SORT.md docs/CLAUDE-CODE-PROMPTS.md \
  docs/HIIT-Station-Operations-Manual.md docs/create_lists.py \
  docs/generate_manual.py docs/list_trello_boards.py docs/mindbody_briefing.py
```

Do NOT delete the gitignored `docs/*.docx` or `hiit-bot-*.json` (see Global Constraints).

- [ ] **Step 4: Rewrite `requirements.txt`**

Replace the whole file with:

```
requests==2.32.3
flask==3.1.0
twilio==9.4.3
anthropic==0.42.0
gunicorn==23.0.0
```

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest -q`
Expected: all pass, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/test_mindbody_only.py
git commit -m "chore: remove non-MindBody helpers, content calendar, docs and deps

All removed files are recoverable from tag pre-mindbody-only.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Rewrite `CLAUDE.md` and verify end to end

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: the final state from Tasks 3–5.
- Produces: accurate project docs; verified branch ready for the user to review and deploy.

- [ ] **Step 1: Replace `CLAUDE.md`**

Replace the whole file with:

````markdown
# HIIT Station Capalaba — WhatsApp Bot

## What this project is

A Flask-based WhatsApp bot for HIIT Station Capalaba gym that answers questions about MindBody data only (classes, members, payments, revenue, reports). Receives messages via Twilio webhook, processes them with the Claude API (tool use), and replies via WhatsApp. Deployed on Railway. Anything outside MindBody gets a one-line refusal.

## Repo structure

```
app.py                — Flask app: Twilio webhook, users, system prompt, tool definitions and routing
mindbody_helper.py    — MindBody API v6 integration and WhatsApp formatters
tests/                — unittest suites (run with `python3 -m pytest -q`)
docs/list_contracts.py — one-off: list every contract product (`railway run python docs/list_contracts.py`)
docs/superpowers/     — design specs and implementation plans
```

Code removed in the MindBody-only cleanup (Outlook, Gmail, Google Calendar, Trello, JotForm, lead automation, Neuform content calendar) is recoverable from git tag `pre-mindbody-only`.

## Architecture

```
WhatsApp → Twilio → /webhook (app.py) → Claude API → MindBody tool calls → reply via Twilio REST
```

Routes: `/webhook`, `/health`, `/`.

## Deployment

- **Platform:** Railway (project: adequate-playfulness, service: hiit-automations)
- **Deploy command:** `railway up --detach` from this directory
- **Runtime:** Python 3.11 (`runtime.txt`) with gunicorn (see Procfile / railway.toml)
- **Environment variables:** set in the Railway dashboard, never in code

| Variable | Purpose |
|----------|---------|
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM` | Twilio WhatsApp |
| `ANTHROPIC_API_KEY` | Claude API |
| `MINDBODY_API_KEY`, `MINDBODY_SOURCE_NAME`, `MINDBODY_SOURCE_PASSWORD`, `MINDBODY_SITE_ID` | MindBody API v6 |
| `APPROVED_NUMBERS` | Comma-separated phone numbers allowed to use the bot |
| `REVENUE_PASSWORD` | Password Erin must give before revenue is shown. Unset/blank = Erin gets no revenue |

## Bot tools

- `get_todays_classes` — today's class schedule (names, times, instructors, booking counts)
- `get_classes_history` — class schedule for a date range (past/future)
- `get_daily_briefing` — today's classes and bookings
- `search_clients` — search members by name/email/phone
- `get_client_detail` — one member's membership, tenure and attendance
- `get_member_stats` — active, suspended, expired snapshot
- `get_payment_failures` — failed/declined transactions
- `get_revenue` — last Mon–Sun membership revenue
- `get_new_members` — new signups (7d or 30d)
- `get_membership_movement` — signups and cancellations by month (`days_back`) or for a date range (`start_date`/`end_date`); output is returned verbatim
- `get_arrears_report` — failed payments grouped by client (Thursday report)
- `get_weekly_summary` — weekly wrap-up (Friday report)
- `run_class_report` — new/intro client check for a class or all classes on a day
- `get_noshow_report` — clients booked but not signed in (post-class)

## Users

| Name | Phone | Notes |
|------|-------|-------|
| Sam | +61420233508 | Full access |
| Chonnie | +61481123186 | Full access |
| Erin | +61421188443 | Revenue requires `REVENUE_PASSWORD` (model-enforced) |

Known gap: `get_weekly_summary` includes revenue and is not password-gated for Erin.

## Coding conventions

- Python 3, no type annotations used
- Lazy imports inside `handle_tool_call` (avoids loading modules on startup)
- Thread-safe: token caching and rate limiting use threading locks
- MindBody data is cached (`_cache_get`/`_cache_set`) — 6 days for membership data, 1 hour for classes
- Tool results are formatted as WhatsApp-friendly text (bold with `*`, bullet points)
- `_find_class()` and `_get_class_visits()` are shared helpers for class-based reports
- `_paginated_get()` handles all MindBody pagination
- `_client_name()` extracts full name from any MindBody client dict
- `_normalize_time()` converts any time format (6am, 6:00 PM, 18:00) to HH:MM 24hr

## Important patterns

- If you add a tool, add it to `_MINDBODY_TOOLS`, add a branch in `handle_tool_call`, add it to `EXPECTED_TOOLS` in `tests/test_mindbody_only.py`, and update `SYSTEM_PROMPT` or the model may not use it
- Slow requests get an instant acknowledgment message before processing (see `SLOW_KEYWORDS`)
- The tool loop has a runtime correction: if the model calls the wrong tool for a no-show request, a hint is appended to the tool result redirecting it
- Long WhatsApp replies are split into ≤1500-char messages at paragraph breaks
- "refresh" / "clear cache" clears the MindBody cache; "/reset" clears chat history
````

- [ ] **Step 2: Full test suite**

Run: `python3 -m pytest -q`
Expected: all pass, 0 failures.

- [ ] **Step 3: Grep for leftovers**

Run: `git grep -nEi 'gcal_helper|gmail_helper|outlook_helper|trello_helper|jotform_helper|lead_automation|samistheman|_get_tools_for_user|USER_TRELLO|USER_GMAIL|USER_CALENDAR|USER_EMAILS|/cron/leads|/api/drive' -- ':!docs/superpowers' ':!tests/test_mindbody_only.py' || echo CLEAN`
Expected: only mentions inside `CLAUDE.md`'s "recoverable from tag" sentence would be acceptable, but that sentence names no modules, so expected output is `CLEAN`.

- [ ] **Step 4: Clean-venv import check (mirrors the Railway build)**

```bash
VENV="$(mktemp -d)/venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -r requirements.txt
"$VENV/bin/python" -c "import app; print(sorted(r.rule for r in app.app.url_map.iter_rules() if r.endpoint != 'static'))"
```

Expected: last line prints `['/', '/health', '/webhook']` with no ImportError.

- [ ] **Step 5: Local webhook smoke test (no network, no deploy)**

```bash
APPROVED_NUMBERS= TWILIO_AUTH_TOKEN= python3 -c "
import app
c = app.app.test_client()
print(c.get('/health').status_code, c.get('/health').data)
r = c.post('/webhook', data={'Body': 'hi', 'From': 'whatsapp:+61400000000'})
print(r.status_code, r.data)
"
```

Expected: `200 b'OK'` then `200` with a TwiML body containing `Sorry, I am not able to help with that.` (unapproved number is rejected before any Claude/MindBody call).

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: rewrite CLAUDE.md for the MindBody-only bot

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Report to the user (no push, no deploy)**

Tell the user:
- Branch `chore/mindbody-only` is ready for review; tag `pre-mindbody-only` holds the archive. Nothing pushed or deployed.
- Before deploying: set `REVENUE_PASSWORD` in Railway (otherwise Erin gets no revenue at all).
- Env vars they can delete from Railway: `MS_CLIENT_ID`, `MS_CLIENT_SECRET`, `MS_TENANT_ID`, `OUTLOOK_USER`, `GOOGLE_SERVICE_ACCOUNT_JSON`, `GOOGLE_CALENDAR_ID`, `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `GMAIL_TOKENS`, `TRELLO_API_KEY`, `TRELLO_TOKEN`, `JOTFORM_API_KEY`, `NEUFORM_UPLOAD_KEY`, `CRON_SECRET`.
- If an external cron service calls `/cron/leads`, it will now get 404 — disable it.
- The Neuform content calendar page's Drive uploads stop working once this is deployed.
- Left on disk, untracked and not archived: `docs/HIIT-Operations-Manual.docx`, `docs/HIIT-Station-Operations-Manual.docx`, `hiit-bot-*.json` (service account key — can be deleted by the user once the Google env vars are gone).
- Follow-up (not fixed): `get_weekly_summary` shows revenue to Erin without the password.
