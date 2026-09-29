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
