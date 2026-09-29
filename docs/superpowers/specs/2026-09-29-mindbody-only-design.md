# MindBody-Only WhatsApp Bot

**Date:** 2026-09-29
**Status:** Draft — awaiting user review

## Problem

The repo has grown into three projects in one folder: the HIIT WhatsApp bot, the Neuform content calendar, and a pile of one-off scripts and docs. The bot itself also answers questions about Outlook, Gmail, Google Calendar, Trello and JotForm, runs a daily lead-email automation, and serves Google Drive upload endpoints for the content calendar. Several of the files the bot imports (`gcal_helper.py`, `gmail_helper.py`, `outlook_helper.py`, `lead_automation.py`, `requirements.txt`, `runtime.txt`) have never been committed — Railway only works because `railway up` uploads the local folder.

## Goal

This repo is the HIIT Station WhatsApp bot, and the bot answers MindBody questions only. Everything else is removed (not moved elsewhere).

## Non-goals

- Restructuring what remains (splitting `app.py`, turning `mindbody_helper.py` into a package, adding webhook/dispatch tests). Deferred; revisit once the codebase is smaller.
- Changing the behaviour of any MindBody tool or report.
- Changing Railway settings or deploying. The user removes obsolete env vars in the dashboard and decides when to deploy.
- Real authentication for revenue access. The password check stays model-enforced; only its storage changes.

## Sequence

1. **Finish the current branch** (`feat/membership-movement`): commit the pending cancellation-dedup fix in `mindbody_helper.py` and its test in `tests/test_membership_movement.py`; run the suite.
2. **Archive**: commit every currently untracked file as-is (`content-calendar/`, `docs/*`, `gcal_helper.py`, `gmail_helper.py`, `outlook_helper.py`, `lead_automation.py`, `requirements.txt`, `runtime.txt`, `CLAUDE.md`, `.railwayignore`) plus the pending deletion of root `index.html` / `neuform-content-calendar.html`, and tag the commit `pre-mindbody-only`. `.gitignore` continues to exclude the service-account JSON and `.docx` files.
3. **Strip** on a new branch `chore/mindbody-only`, branched from the archive commit.

## What is removed

### Files
- `gcal_helper.py`, `gmail_helper.py`, `outlook_helper.py`, `trello_helper.py`, `jotform_helper.py`, `lead_automation.py`
- `content-calendar/` (entire directory)
- `tests/test_trello_write.py`, `tests/test_jotform_helper.py`
- `.railwayignore` (its only purpose is excluding `content-calendar/`, `docs/`, `*.html`)
- `docs/` except `docs/superpowers/` and `docs/list_contracts.py` (a MindBody maintenance script). Removed: `CLAUDE-CODE-PROMPT-OUTLOOK-SORT.md`, `CLAUDE-CODE-PROMPTS.md`, `HIIT-Station-Operations-Manual.md`, `create_lists.py`, `generate_manual.py`, `list_trello_boards.py`, `mindbody_briefing.py`, and the local (gitignored) `.docx` files.

### `app.py`
- **Tool definitions:** `_CALENDAR_TOOLS`, `_OUTLOOK_TOOLS`, `_GMAIL_TOOLS`, `_TRELLO_WRITE_TOOLS`, `_JOTFORM_TOOLS`, and `get_trello_tasks` (currently inside `_MINDBODY_TOOLS`).
- **Tool routing:** every non-MindBody branch in `handle_tool_call`.
- **Per-user tool filtering:** `_get_tools_for_user`, `_get_user_calendar_ids`, `_get_user_gmail`, `ALL_TOOLS` composition. All users get the same MindBody tool list.
- **Config:** `USER_EMAILS`, `USER_GMAIL`, `USER_CALENDAR`, `USER_TRELLO`. The `user_email` parameter threaded through `handle_tool_call` / `get_claude_response` is removed (only non-MindBody tools used it).
- **Routes:** `/oauth/callback`, `/cron/leads`, `/cron/leads/debug`, `/api/drive/upload`, `/api/drive/folders`.
- **Background scheduler:** `_run_daily_leads`, `start_scheduler`, and its startup calls.
- **Commands:** "connect gmail" / "setup gmail" / "link gmail" handling in `process_message_async`.
- **`SLOW_KEYWORDS`:** `"trello"`, `"hiit challenge"`, `"tasks"`, `"submission"`, `"submissions"`, `"jotform"`, `"form submissions"`.
- **System prompt:** Chonnie's two-email-accounts addition and the calendar-event creation rules in `_build_system_prompt`.

### Dependencies (`requirements.txt`)
`msal`, `google-api-python-client`, `google-auth`, `python-docx`.

## What stays

- Routes: `/webhook`, `/health`, `/`.
- Twilio validation, approved-number check, rate limiting, conversation history, `/reset`, instant acknowledgments for slow requests, 1500-char truncation, `max_tokens=512`.
- `USER_NAMES`, `APPROVED_NUMBERS`.
- `mindbody_helper.py` unchanged.
- `tests/test_membership_movement.py`.
- MindBody tools (14): `get_todays_classes`, `get_daily_briefing`, `search_clients`, `get_client_detail`, `get_member_stats`, `get_payment_failures`, `get_classes_history`, `get_revenue`, `get_new_members`, `get_membership_movement`, `get_arrears_report`, `get_weekly_summary`, `run_class_report`, `get_noshow_report`.
- The runtime no-show correction in the tool loop.

## What changes

### Daily briefing
`get_daily_briefing` returns classes and bookings only (the `format_briefing` output). Its tool description changes from "classes, bookings, calendar, inbox" to classes and bookings.

### System prompt
- Opening scoped to MindBody: the assistant answers questions about HIIT Station Capalaba's MindBody data (classes, members, payments, revenue, reports).
- New rule: if the user asks for something outside MindBody (email, calendar, Trello, forms, general tasks), reply in one line that this bot only handles MindBody questions. Do not attempt it.
- Kept verbatim: the verbatim-output rule for `get_membership_movement`, `get_client_detail` vs `search_clients` routing, membership-movement date handling, no-show routing, and the formatting rules.
- Removed: the Trello, Google Calendar, Outlook, email-draft, and JotForm guidance.

### Revenue password
- Read from env var `REVENUE_PASSWORD` instead of the literal in `_build_system_prompt`.
- If `REVENUE_PASSWORD` is unset or empty, fail closed: Erin's prompt says revenue data is not available to this user and `get_revenue` must not be called; no password is offered.
- The user sets `REVENUE_PASSWORD` in Railway before deploying.

### `CLAUDE.md`
Rewritten to describe the MindBody-only bot: architecture, the 14 tools, users, env vars (including `REVENUE_PASSWORD`), conventions. The repo-structure section drops `content-calendar/` and the non-MindBody helpers.

## Env vars the user can delete from Railway

`MS_CLIENT_ID`, `MS_CLIENT_SECRET`, `MS_TENANT_ID`, `GOOGLE_SERVICE_ACCOUNT_JSON`, `GOOGLE_CALENDAR_ID`, `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `GMAIL_TOKENS`, `TRELLO_API_KEY`, `TRELLO_TOKEN`, and any JotForm / Drive vars found in the removed code (the plan will list exact names by grepping `os.environ` in the removed files).

## Verification

- `python3 -m pytest -q` passes (membership-movement tests, plus new tests for the revenue-password prompt: set, and unset/fail-closed).
- `python3 -c "import app"` succeeds with the removed modules gone.
- `grep` finds no references in the remaining code to `gcal_helper`, `gmail_helper`, `outlook_helper`, `trello_helper`, `jotform_helper`, `lead_automation`, `samistheman`, or the removed tool names.
- Every tool name in the tool list has a `handle_tool_call` branch, and every branch has a tool definition (checked by a small test).
- No deploy.

## Rollback

Everything removed is recoverable from tag `pre-mindbody-only`. The strip lives on its own branch until the user merges it.
