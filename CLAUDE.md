# HIIT Station Capalaba — WhatsApp Bot

## What this project is

A Flask-based WhatsApp bot for HIIT Station Capalaba gym. Receives messages via Twilio webhook, processes them with the Claude API (tool use), and replies via WhatsApp. Deployed on Railway.

## Repo structure

```
/                     — HIIT Bot (deploys to Railway from root)
/content-calendar/    — Neuform Content Calendar (deploys to GitHub Pages, has its own CLAUDE.md)
/docs/                — Reference docs, one-off scripts, operations manuals
```

## Architecture

```
WhatsApp → Twilio → /webhook (app.py) → Claude API (sonnet) → tool calls → reply via Twilio REST
```

- **app.py** — Main Flask app, Twilio webhook, Claude API integration, tool definitions and routing
- **mindbody_helper.py** — MindBody API v6 integration (classes, members, payments, revenue, reports)
- **gcal_helper.py** — Google Calendar via service account (read events, create events)
- **outlook_helper.py** — Microsoft Graph API for Outlook email (read inbox, create drafts)
- **gmail_helper.py** — Gmail OAuth2 integration (read inbox/drafts for specific users)
- **trello_helper.py** — Trello API for HIIT Challenge board tasks

## Deployment

- **Platform:** Railway (project: adequate-playfulness, service: hiit-automations)
- **Deploy command:** `railway up --detach` from this directory
- **Runtime:** Python 3.x with gunicorn (see Procfile / railway.toml)
- **Environment variables:** All API keys/secrets are set in Railway dashboard, not in code

## Key API connections

| Service | Auth method | Config |
|---------|-------------|--------|
| MindBody API v6 | API key + source credentials → user token | `MINDBODY_API_KEY`, `MINDBODY_SOURCE_NAME`, `MINDBODY_SOURCE_PASSWORD`, `MINDBODY_SITE_ID` |
| Twilio (WhatsApp) | Account SID + Auth Token | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM` |
| Claude API | API key | `ANTHROPIC_API_KEY` |
| Microsoft Graph (Outlook) | MSAL client credentials | `MS_CLIENT_ID`, `MS_CLIENT_SECRET`, `MS_TENANT_ID` |
| Google Calendar | Service account JSON | `GOOGLE_SERVICE_ACCOUNT_JSON`, `GOOGLE_CALENDAR_ID` |
| Gmail | OAuth2 (per-user refresh tokens) | `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `GMAIL_TOKENS` |
| Trello | API key + token | `TRELLO_API_KEY`, `TRELLO_TOKEN` |

## Bot tools (what the WhatsApp bot can do)

### MindBody
- `get_todays_classes` — today's class schedule (names, times, instructors, booking counts)
- `get_classes_history` — class schedule for a date range (past/future)
- `get_daily_briefing` — full briefing: classes + calendar + inbox
- `search_clients` — search members by name/email/phone
- `get_member_stats` — active members, suspended, cancellations, new signups
- `get_payment_failures` — failed/declined transactions
- `get_revenue` — last Mon-Sun membership revenue
- `get_new_members` — new signups (7d or 30d)
- `get_arrears_report` — failed payments grouped by client (Thursday report)
- `get_weekly_summary` — weekly wrap-up (Friday report)
- `run_class_report` — new/intro client check for a specific class
- `get_noshow_report` — clients booked but not signed in (post-class)

### Google Calendar
- `get_calendar_events` — read events for a date range
- `create_calendar_event` — create a new event

### Outlook
- `read_inbox` — read recent emails
- `draft_email` — create a draft (never sends directly)

### Gmail (Chonnie only)
- `read_gmail` — read Gmail inbox
- `read_gmail_drafts` — read Gmail drafts

### Trello
- `get_trello_tasks` — HIIT Challenge board cards due today or overdue

## Users

| Name | Phone | Email | Special access |
|------|-------|-------|----------------|
| Sam | +61420233508 | sam@hiitaustralia.com.au | Full access |
| Chonnie | +61481123186 | chontel@hiitaustralia.com.au | Has Gmail (chontelhiit@gmail.com) + Outlook |
| Erin | +61421188443 | admin@hiitaustralia.com.au | Revenue is password-protected |

## Coding conventions

- Python 3, no type annotations used
- Lazy imports inside `handle_tool_call` (avoids loading all modules on startup)
- Thread-safe: token caching and rate limiting use threading locks
- MindBody data is cached (`_cache_get`/`_cache_set`) — 6 days for membership data, 1 hour for classes
- Tool results are formatted as WhatsApp-friendly text (bold with `*`, bullet points)
- `_find_class()` and `_get_class_visits()` are shared helpers for class-based reports
- `_paginated_get()` handles all MindBody pagination
- `_client_name()` extracts full name from any MindBody client dict
- `_normalize_time()` converts any time format (6am, 6:00 PM, 18:00) to HH:MM 24hr
- Per-user tool filtering: only send tools relevant to each user to save API tokens

## Important patterns

- The system prompt explicitly lists what tools to use for what — if you add a new tool, also update `SYSTEM_PROMPT` in app.py or the model may not use it
- Slow requests get an instant acknowledgment message before processing (see `SLOW_KEYWORDS`)
- The tool loop has a runtime correction: if the model calls the wrong tool for a no-show request, a hint is appended to the tool result redirecting it
- WhatsApp replies are truncated to 1500 chars
- `max_tokens=512` on Claude API calls to save cost
