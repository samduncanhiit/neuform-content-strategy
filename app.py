"""
WhatsApp Bot for HIIT Station Capalaba
Receives messages via Twilio webhook, processes with Claude API, responds via WhatsApp.
"""

import os
import sys
import logging
import threading
import time
import requests
from collections import defaultdict
from functools import wraps

from flask import Flask, request, abort
from twilio.rest import Client as TwilioClient
from twilio.request_validator import RequestValidator
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# Set up logging so Railway captures output
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
TWILIO_WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

# Comma-separated list of approved phone numbers (e.g. "+61420233508,+61400000000")
APPROVED_NUMBERS_RAW = os.environ.get("APPROVED_NUMBERS", "")
APPROVED_NUMBERS = {
    n.strip() for n in APPROVED_NUMBERS_RAW.split(",") if n.strip()
}

USER_NAMES = {
    "+61420233508": "Sam",
    "+61481123186": "Chonnie",
    "+61421188443": "Erin",
}

USER_EMAILS = {
    "+61420233508": "sam@hiitaustralia.com.au",
    "+61481123186": "chontel@hiitaustralia.com.au",
    "+61421188443": "admin@hiitaustralia.com.au",
}

# Gmail accounts (separate from Outlook)
USER_GMAIL = {
    "+61481123186": "chontelhiit@gmail.com",
}

# Google Calendar IDs per user
USER_CALENDAR = {
    "+61420233508": [os.environ.get("GOOGLE_CALENDAR_ID", "primary")],
    "+61481123186": ["chontelhiit@gmail.com"],
    "+61421188443": [
        "hiitstationmealplan@gmail.com",
        "2edc99a76751fd3e9b49e8b5ad1e88c3b008086eb5fe90ce8554dcea6910b4e1@group.calendar.google.com",
        "cd723d3f7abb0bd192760dbe566891837b4bf3fda386c45628c80c1daedf5ecd@group.calendar.google.com",
    ],
}

SYSTEM_PROMPT = (
    "You are an AI assistant for HIIT Station Capalaba helping manage daily operations. "
    "You are helpful, concise, and professional. You assist with scheduling, member queries, "
    "class information, and general operational tasks. Keep responses brief and suitable "
    "for WhatsApp messaging.\n\n"
    "You have access to tools for MindBody (classes, members, revenue, payments, "
    "new member signups, arrears report, weekly summary, no-show reports), "
    "Trello (HIIT Challenge board tasks), "
    "Google Calendar (personal calendar events and meetings), and "
    "Outlook (reading inbox, creating email drafts). "
    "IMPORTANT: When the user asks about 'my calendar', 'meetings', 'appointments', or 'what do I have on', "
    "always use the get_calendar_events tool (Google Calendar), NOT MindBody classes. "
    "MindBody is only for gym class schedules — use get_todays_classes or get_classes_history for that. "
    "IMPORTANT: When the user asks about no-shows, who didn't show up, who didn't sign in, "
    "or who didn't attend a class, you MUST use the get_noshow_report tool. "
    "Do NOT use get_todays_classes or get_classes_history for this — those only show booking counts. "
    "get_noshow_report checks the actual sign-in roster and returns individual client names. "
    "When the user asks to send an email, always create a draft instead — never send directly. "
    "FORMATTING: All responses must be plain text suitable for copy-pasting into other chats. "
    "Never use markdown tables, horizontal lines (---), pipes (|), or special formatting. "
    "Use simple lists with numbers or bullet points. Use *bold* for headings only. "
    "Keep it clean and easy to copy-paste."
)

# ── Security: Rate limiter ────────────────────────────────────────────────────

_rate_limit_lock = threading.Lock()
_request_timestamps = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 10


def is_rate_limited(sender):
    """Check if a sender has exceeded the rate limit."""
    now = time.time()
    with _rate_limit_lock:
        timestamps = _request_timestamps[sender]
        # Remove timestamps older than 60 seconds
        _request_timestamps[sender] = [t for t in timestamps if now - t < 60]
        if len(_request_timestamps[sender]) >= MAX_REQUESTS_PER_MINUTE:
            return True
        _request_timestamps[sender].append(now)
        return False


# ── Security: Twilio signature validation ─────────────────────────────────────

_twilio_validator = None


def get_twilio_validator():
    global _twilio_validator
    if _twilio_validator is None and TWILIO_AUTH_TOKEN:
        _twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)
    return _twilio_validator


def validate_twilio_request(f):
    """Decorator to verify incoming requests are from Twilio."""
    @wraps(f)
    def decorated(*args, **kwargs):
        validator = get_twilio_validator()
        if validator is None:
            logger.warning("Twilio auth token not set — skipping signature validation")
            return f(*args, **kwargs)

        signature = request.headers.get("X-Twilio-Signature", "")
        # Railway runs behind a reverse proxy — Flask sees http:// but Twilio
        # calculates the signature using the public https:// URL
        url = request.url.replace("http://", "https://", 1)
        post_vars = request.form.to_dict()

        if not validator.validate(url, post_vars, signature):
            logger.warning(f"Invalid Twilio signature — rejecting request")
            abort(403)

        return f(*args, **kwargs)
    return decorated


# ── Lazy client initialization ────────────────────────────────────────────────

_claude_client = None
_claude_lock = threading.Lock()


def get_claude_client():
    global _claude_client
    with _claude_lock:
        if _claude_client is None:
            import anthropic
            _claude_client = anthropic.Anthropic(
                api_key=ANTHROPIC_API_KEY,
                timeout=60.0,
            )
    return _claude_client


logger.info(f"App loaded. {len(APPROVED_NUMBERS)} approved numbers configured")
logger.info(f"ANTHROPIC_API_KEY set: {bool(ANTHROPIC_API_KEY)}")
logger.info(f"TWILIO_ACCOUNT_SID set: {bool(TWILIO_ACCOUNT_SID)}")


# ── Tools ─────────────────────────────────────────────────────────────────────

MAX_DAYS_BACK = 90

# ── Tool definitions ──────────────────────────────────────────────────────────
# Organised by category so we can send only the tools each user needs.

_MINDBODY_TOOLS = [
    {
        "name": "get_todays_classes",
        "description": "Get today's class schedule with names, times, instructors, bookings",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_daily_briefing",
        "description": "Full daily briefing: classes, bookings, calendar, inbox",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_clients",
        "description": "Search MindBody clients by name, email, or phone",
        "input_schema": {
            "type": "object",
            "properties": {
                "search_text": {"type": "string", "description": "Name, email, or phone"},
            },
            "required": ["search_text"],
        },
    },
    {
        "name": "get_member_stats",
        "description": "Membership stats: active, suspended, cancellations (7d), expired, new signups. When user asks about 'cancellations' use this, NOT class cancellations.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_payment_failures",
        "description": "Payment failures and transaction summary (last 30 days)",
        "input_schema": {
            "type": "object",
            "properties": {
                "days_back": {"type": "integer", "description": "Days back (default 30, max 90)", "default": 30},
            },
            "required": [],
        },
    },
    {
        "name": "get_classes_history",
        "description": "Class schedule for a date range (past and/or future)",
        "input_schema": {
            "type": "object",
            "properties": {
                "days_back": {"type": "integer", "description": "Days back (default 0, max 90)", "default": 0},
                "days_forward": {"type": "integer", "description": "Days forward (default 0, max 7)", "default": 0},
            },
            "required": [],
        },
    },
    {
        "name": "get_revenue",
        "description": "Membership debit revenue for the last Mon-Sun week. Only report what this tool returns.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_new_members",
        "description": "New member sign-ups (last 7 or 30 days) with name, date, membership, email, phone",
        "input_schema": {
            "type": "object",
            "properties": {
                "days_back": {"type": "integer", "description": "7 or 30 (default 7)", "default": 7},
            },
            "required": [],
        },
    },
    {
        "name": "get_arrears_report",
        "description": "Failed/declined payments (30 days) grouped by client with total owed",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_weekly_summary",
        "description": "Weekly wrap-up: classes, bookings, revenue, signups, cancellations (last Mon-Sun)",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_trello_tasks",
        "description": "HIIT Challenge Trello cards due today or overdue",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_class_report",
        "description": "New client report on a class: first-timers, intro/trial pricing, new memberships (14d)",
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string", "description": "Class name only, e.g. 'HIIT Rox', 'HIIT Maxx'. Do NOT include the time here."},
                "class_date": {"type": "string", "description": "YYYY-MM-DD (default today)"},
                "class_time": {"type": "string", "description": "Time to disambiguate, e.g. '6am', '18:00', '6:00 PM'"},
            },
            "required": ["class_name"],
        },
    },
    {
        "name": "get_noshow_report",
        "description": "No-show report: clients booked but not signed in after class finished. Use when user asks 'who didn't show up', 'no shows', 'didn't sign in'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string", "description": "Class name only, e.g. 'HIIT Rox', 'HIIT Maxx'. Do NOT include the time here."},
                "class_date": {"type": "string", "description": "YYYY-MM-DD (default today)"},
                "class_time": {"type": "string", "description": "Time to disambiguate, e.g. '6am', '18:00', '6:00 PM'"},
            },
            "required": ["class_name"],
        },
    },
]

_CALENDAR_TOOLS = [
    {
        "name": "get_calendar_events",
        "description": "Get Google Calendar events for today or a date range",
        "input_schema": {
            "type": "object",
            "properties": {
                "days_forward": {"type": "integer", "description": "Days forward (default 0, max 7)", "default": 0},
                "days_back": {"type": "integer", "description": "Days back (default 0)", "default": 0},
            },
            "required": [],
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new Google Calendar event",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Event title"},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "start_time": {"type": "string", "description": "HH:MM 24hr start"},
                "end_time": {"type": "string", "description": "HH:MM 24hr end"},
                "description": {"type": "string", "description": "Optional description"},
                "location": {"type": "string", "description": "Optional location"},
            },
            "required": ["summary", "start_date", "start_time", "end_time"],
        },
    },
]

_OUTLOOK_TOOLS = [
    {
        "name": "read_inbox",
        "description": "Read recent Outlook emails",
        "input_schema": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Emails to fetch (default 10, max 50)", "default": 10},
                "search": {"type": "string", "description": "Optional search query"},
            },
            "required": [],
        },
    },
    {
        "name": "draft_email",
        "description": "Create an Outlook draft email (saved, not sent)",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient(s), comma-separated"},
                "subject": {"type": "string", "description": "Subject line"},
                "body": {"type": "string", "description": "Email body"},
                "cc": {"type": "string", "description": "CC address(es), comma-separated"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]

_GMAIL_TOOLS = [
    {
        "name": "read_gmail",
        "description": "Read recent Gmail emails",
        "input_schema": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Emails to fetch (default 10)", "default": 10},
                "query": {"type": "string", "description": "Optional search query"},
            },
            "required": [],
        },
    },
    {
        "name": "read_gmail_drafts",
        "description": "Read Gmail drafts",
        "input_schema": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Drafts to fetch (default 10)", "default": 10},
            },
            "required": [],
        },
    },
]

# Users who have Gmail access — only they get Gmail tools
_GMAIL_USERS = set(USER_GMAIL.values())


def _get_tools_for_user(user_email):
    """Return only the tools relevant to this user — saves ~500 input tokens for non-Gmail users."""
    tools = _MINDBODY_TOOLS + _CALENDAR_TOOLS + _OUTLOOK_TOOLS
    # Only include Gmail tools for users who actually have a Gmail account
    if user_email and any(user_email == USER_EMAILS.get(phone) for phone in USER_GMAIL):
        tools = tools + _GMAIL_TOOLS
    return tools


# Keep ALL_TOOLS for handle_tool_call routing (it handles all tools regardless)
ALL_TOOLS = _MINDBODY_TOOLS + _CALENDAR_TOOLS + _OUTLOOK_TOOLS + _GMAIL_TOOLS


def _get_user_calendar_ids(user_email):
    """Look up per-user calendar IDs from the phone→email→calendar mapping."""
    for phone, cals in USER_CALENDAR.items():
        if user_email and user_email == USER_EMAILS.get(phone):
            return cals
    return ["primary"]


def _get_user_gmail(user_email):
    """Look up the Gmail address for a user from the phone→email→Gmail mapping."""
    for phone, gmail in USER_GMAIL.items():
        if user_email and user_email == USER_EMAILS.get(phone):
            return gmail
    return user_email


def handle_tool_call(tool_name, tool_input, user_email=None):
    """Execute a tool call and return the result."""
    # ── MindBody tools ────────────────────────────────────────────────────────
    if tool_name == "get_todays_classes":
        from mindbody_helper import get_todays_schedule, format_schedule
        classes = get_todays_schedule()
        return format_schedule(classes)

    elif tool_name == "get_daily_briefing":
        from mindbody_helper import get_daily_briefing, format_briefing
        from gcal_helper import get_events, format_events
        from outlook_helper import read_inbox, format_inbox_summary

        parts = []

        # Classes
        briefing = get_daily_briefing()
        parts.append(format_briefing(briefing))

        # Calendar — use per-user calendar IDs (may be multiple)
        try:
            all_events = []
            for cid in _get_user_calendar_ids(user_email):
                try:
                    all_events.extend(get_events(days_forward=0, days_back=0, calendar_id=cid))
                except Exception:
                    pass
            all_events.sort(key=lambda e: e.get("start", ""))
            parts.append(format_events(all_events, title="TODAY'S CALENDAR"))
        except Exception:
            parts.append("*TODAY'S CALENDAR*\nUnable to fetch calendar.")

        # Emails
        try:
            emails = read_inbox(count=5, outlook_user=user_email)
            parts.append(format_inbox_summary(emails))
        except Exception:
            parts.append("*INBOX*\nUnable to fetch emails.")

        return "\n\n".join(parts)

    elif tool_name == "search_clients":
        from mindbody_helper import search_clients, format_clients
        search_text = tool_input.get("search_text", "")[:100]  # Cap search length
        clients = search_clients(search_text)
        return format_clients(clients)

    elif tool_name == "get_member_stats":
        from mindbody_helper import get_member_stats, format_member_stats
        stats = get_member_stats()
        return format_member_stats(stats)

    elif tool_name == "get_payment_failures":
        from mindbody_helper import get_payment_failures, format_payment_failures
        days = min(tool_input.get("days_back", 30), MAX_DAYS_BACK)
        payments = get_payment_failures(days_back=days)
        return format_payment_failures(payments)

    elif tool_name == "get_revenue":
        from mindbody_helper import get_revenue, format_revenue
        rev = get_revenue()
        return format_revenue(rev)

    elif tool_name == "get_classes_history":
        from mindbody_helper import get_classes, format_schedule
        days_back = min(tool_input.get("days_back", 0), MAX_DAYS_BACK)
        days_forward = min(tool_input.get("days_forward", 0), 7)
        classes = get_classes(days_back=days_back, days_forward=days_forward)
        return format_schedule(classes)

    elif tool_name == "get_new_members":
        from mindbody_helper import get_new_members, format_new_members
        days = tool_input.get("days_back", 7)
        if days not in (7, 30):
            days = 7
        members = get_new_members(days_back=days)
        return format_new_members(members, days_back=days)

    elif tool_name == "get_arrears_report":
        from mindbody_helper import get_arrears_report, format_arrears_report
        report = get_arrears_report()
        return format_arrears_report(report)

    elif tool_name == "get_weekly_summary":
        from mindbody_helper import get_weekly_summary, format_weekly_summary
        summary = get_weekly_summary()
        return format_weekly_summary(summary)

    elif tool_name == "run_class_report":
        from mindbody_helper import run_class_report, format_class_report
        report = run_class_report(
            class_name=tool_input["class_name"],
            class_date=tool_input.get("class_date"),
            class_time=tool_input.get("class_time"),
        )
        return format_class_report(report)

    elif tool_name == "get_noshow_report":
        from mindbody_helper import get_noshow_report, format_noshow_report
        report = get_noshow_report(
            class_name=tool_input["class_name"],
            class_date=tool_input.get("class_date"),
            class_time=tool_input.get("class_time"),
        )
        return format_noshow_report(report)

    elif tool_name == "get_trello_tasks":
        from trello_helper import get_trello_tasks, format_trello_tasks
        tasks = get_trello_tasks()
        return format_trello_tasks(tasks)

    # ── Google Calendar tools ─────────────────────────────────────────────────
    elif tool_name == "get_calendar_events":
        from gcal_helper import get_events, format_events
        days_fwd = min(tool_input.get("days_forward", 0), 7)
        days_back = min(tool_input.get("days_back", 0), 7)
        all_events = []
        for cid in _get_user_calendar_ids(user_email):
            try:
                all_events.extend(get_events(days_forward=days_fwd, days_back=days_back, calendar_id=cid))
            except Exception:
                pass
        # Sort by start time
        all_events.sort(key=lambda e: e.get("start", ""))
        return format_events(all_events, title="Calendar")

    elif tool_name == "create_calendar_event":
        from gcal_helper import create_event
        cal_ids = _get_user_calendar_ids(user_email)
        cal_id = cal_ids[0] if cal_ids != ["primary"] else None
        result = create_event(
            summary=tool_input["summary"],
            start_date=tool_input["start_date"],
            start_time=tool_input["start_time"],
            end_time=tool_input["end_time"],
            description=tool_input.get("description"),
            location=tool_input.get("location"),
            calendar_id=cal_id,
        )
        return f"Event created: {result['summary']}"

    # ── Outlook tools ─────────────────────────────────────────────────────────
    elif tool_name == "read_inbox":
        from outlook_helper import read_inbox, format_inbox_summary
        emails = read_inbox(
            count=min(tool_input.get("count", 10), 50),
            search=tool_input.get("search"),
            outlook_user=user_email,
        )
        return format_inbox_summary(emails)

    elif tool_name == "draft_email":
        from outlook_helper import create_draft
        result = create_draft(
            to=tool_input["to"],
            subject=tool_input["subject"],
            body=tool_input["body"],
            cc=tool_input.get("cc"),
            outlook_user=user_email,
        )
        return f"Draft created: {result['subject']}"

    # ── Gmail tools ─────────────────────────────────────────────────────────
    elif tool_name == "read_gmail":
        from gmail_helper import read_gmail_inbox, format_gmail_inbox
        gmail_addr = _get_user_gmail(user_email)
        emails = read_gmail_inbox(gmail_addr, count=tool_input.get("count", 10), query=tool_input.get("query"))
        return format_gmail_inbox(emails)

    elif tool_name == "read_gmail_drafts":
        from gmail_helper import read_gmail_drafts as fetch_drafts, format_gmail_drafts
        gmail_addr = _get_user_gmail(user_email)
        drafts = fetch_drafts(gmail_addr, count=tool_input.get("count", 10))
        return format_gmail_drafts(drafts)

    return f"Unknown tool: {tool_name}"


def _build_system_prompt(user_name, raw_number):
    """Build the system prompt with user-specific additions."""
    from datetime import datetime, timezone, timedelta
    aest = timezone(timedelta(hours=10))
    today = datetime.now(aest)
    today_str = today.strftime("%A, %B %-d, %Y")

    parts = [SYSTEM_PROMPT]

    if user_name:
        parts.append(f"The user's name is {user_name}. Greet them as 'Hey {user_name}'.")

    # Erin: password-protect revenue data
    if raw_number == "+61421188443":
        parts.append(
            "IMPORTANT: This user does NOT have access to revenue data. "
            "If they ask about revenue, income, money, debits, or financial reports, "
            "ask for a password first. Correct password: 'samistheman'. "
            "Only call get_revenue if they give the exact password."
        )

    # Chonnie has both Outlook and Gmail
    if raw_number == "+61481123186":
        parts.append(
            "This user has two email accounts: "
            "Outlook (chontel@hiitaustralia.com.au, use read_inbox) and "
            "Gmail (chontelhiit@gmail.com, use read_gmail). "
            "When they ask to check emails or drafts, ask which account."
        )

    parts.append(f"Today is {today_str}.")
    parts.append(
        "When user says 'next week' they mean the upcoming Mon-Sun. "
        "Calculate correct dates from today."
    )
    if user_name:
        parts.append(
            "When creating calendar events: create immediately, don't ask follow-ups unless "
            "date/time is genuinely unclear. Default end = 1hr after start. "
            "Ask about description notes AFTER creating. "
            f"Prefix event name with '{user_name} - '."
        )

    return "\n\n".join(parts)


def get_claude_response(user_message, sender=None):
    """Send a message to Claude with tools and return the text response."""
    client = get_claude_client()
    messages = [{"role": "user", "content": user_message}]

    raw_number = (sender or "").replace("whatsapp:", "").strip()
    user_name = USER_NAMES.get(raw_number)
    user_email = USER_EMAILS.get(raw_number, "sam@hiitaustralia.com.au")
    system = _build_system_prompt(user_name, raw_number)
    tools = _get_tools_for_user(user_email)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        tools=tools,
        messages=messages,
    )

    # Detect if the user is asking about no-shows so we can correct wrong tool usage
    _noshow_keywords = ("no show", "no-show", "noshow", "didn't show", "didn't sign in",
                        "not signed in", "didn't attend", "didn't turn up", "who missed")
    user_wants_noshow = any(kw in user_message.lower() for kw in _noshow_keywords)

    # Handle tool use loop
    while response.stop_reason == "tool_use":
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for block in assistant_content:
            if block.type == "tool_use":
                logger.info(f"Tool call: {block.name}")
                try:
                    result = handle_tool_call(block.name, block.input, user_email=user_email)
                except Exception as e:
                    logger.error(f"Tool error ({block.name}): {e}")
                    result = "Sorry, that data is temporarily unavailable."

                # If model used the wrong tool for a no-show request, redirect it
                if user_wants_noshow and block.name in ("get_todays_classes", "get_classes_history"):
                    result += (
                        "\n\nNOTE: This tool only shows booking counts. "
                        "To get the no-show list with individual client names, "
                        "you MUST call the get_noshow_report tool with class_name and class_time."
                    )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages,
        )

    for block in response.content:
        if hasattr(block, "text"):
            return block.text

    return "I processed your request but have no response to show."


def is_approved(phone_number):
    """Check if a phone number is in the approved list."""
    raw = phone_number.replace("whatsapp:", "").strip()
    return raw in APPROVED_NUMBERS


def mask_number(phone_number):
    """Mask phone number for logging — show last 4 digits only."""
    raw = phone_number.replace("whatsapp:", "").strip()
    if len(raw) > 4:
        return f"***{raw[-4:]}"
    return "****"


# ── Routes ─────────────────────────────────────────────────────────────────────


WHATSAPP_CHAR_LIMIT = 4000  # WhatsApp allows 4096, leave a small buffer


def send_whatsapp_reply(to, body):
    """Send a WhatsApp message via Twilio REST API.

    If the message exceeds WhatsApp's limit, splits into multiple messages
    at paragraph boundaries so nothing gets cut off.
    """
    twilio = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    if len(body) <= WHATSAPP_CHAR_LIMIT:
        twilio.messages.create(from_=TWILIO_WHATSAPP_FROM, to=to, body=body)
        logger.info(f"Reply sent to {mask_number(to)} ({len(body)} chars)")
        return

    # Split on double newlines (paragraph breaks) to keep formatting clean
    paragraphs = body.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > WHATSAPP_CHAR_LIMIT:
            if current:
                chunks.append(current)
            current = para
        else:
            current = candidate

    if current:
        chunks.append(current)

    for i, chunk in enumerate(chunks):
        twilio.messages.create(from_=TWILIO_WHATSAPP_FROM, to=to, body=chunk)
        logger.info(f"Reply {i+1}/{len(chunks)} sent to {mask_number(to)} ({len(chunk)} chars)")


# Tools that typically take >10 seconds (membership lookups, revenue, briefings)
SLOW_KEYWORDS = [
    "member", "membership", "active members", "how many",
    "revenue", "debit", "cancel", "cancellation",
    "briefing", "daily briefing", "report",
    "stats", "statistics",
    "new members", "new signups", "sign-ups", "sign ups",
    "arrears", "failed payments", "owed",
    "weekly summary", "weekly wrap", "wrap-up", "wrap up",
    "trello", "hiit challenge", "tasks",
    "class report", "run a report", "check the class", "tonight's class",
    "no show", "no-show", "didn't show", "didn't sign in", "not signed in",
]

QUICK_REPLIES = [
    "Sure, let me get that for you!",
    "On it — pulling that data now!",
    "Done, give me a moment to grab that for you!",
]


def _is_slow_request(msg):
    """Check if a message is likely to trigger a slow API call."""
    msg_lower = msg.lower()
    return any(kw in msg_lower for kw in SLOW_KEYWORDS)


def process_message_async(sender, incoming_msg):
    """Process the message in a background thread and send reply via Twilio API."""
    import random

    # Handle Gmail connect command
    if incoming_msg.lower().strip() in ("connect gmail", "setup gmail", "link gmail"):
        from gmail_helper import get_auth_url
        auth_url = get_auth_url()
        send_whatsapp_reply(sender, f"Click this link to connect your Gmail:\n\n{auth_url}")
        return

    # Handle cache refresh command
    if incoming_msg.lower().strip() in ("refresh", "refresh data", "clear cache"):
        from mindbody_helper import _cache_clear
        _cache_clear()
        send_whatsapp_reply(sender, "Cache cleared! Next request will pull fresh data.")
        return

    # Send instant acknowledgment for slow requests
    if _is_slow_request(incoming_msg):
        try:
            ack = random.choice(QUICK_REPLIES)
            send_whatsapp_reply(sender, ack)
        except Exception:
            pass

    try:
        logger.info("Calling Claude API...")
        reply_text = get_claude_response(incoming_msg, sender=sender)
        logger.info(f"Claude response received ({len(reply_text)} chars)")
    except Exception as e:
        logger.error(f"Claude API error: {type(e).__name__}")
        reply_text = "Sorry, something went wrong. Please try again later."

    try:
        send_whatsapp_reply(sender, reply_text)
    except Exception as e:
        logger.error(f"Failed to send WhatsApp reply: {type(e).__name__}")


@app.route("/webhook", methods=["POST"])
@validate_twilio_request
def webhook():
    """Handle incoming WhatsApp messages from Twilio."""
    incoming_msg = request.form.get("Body", "").strip()
    sender = request.form.get("From", "")

    logger.info(f"Incoming message from {mask_number(sender)}")

    if not incoming_msg:
        resp = MessagingResponse()
        resp.message("I received an empty message. Please try again.")
        return str(resp), 200

    if not is_approved(sender):
        logger.info(f"Rejected unapproved number: {mask_number(sender)}")
        resp = MessagingResponse()
        resp.message("Sorry, I am not able to help with that.")
        return str(resp), 200

    if is_rate_limited(sender):
        logger.warning(f"Rate limited: {mask_number(sender)}")
        resp = MessagingResponse()
        resp.message("You're sending too many messages. Please wait a moment.")
        return str(resp), 200

    # Process in background thread to avoid Twilio's 15-second timeout
    thread = threading.Thread(target=process_message_async, args=(sender, incoming_msg))
    thread.start()

    return "", 200


@app.route("/oauth/callback", methods=["GET"])
def oauth_callback():
    """Handle Google OAuth2 callback for Gmail access."""
    from gmail_helper import exchange_code, store_tokens
    import json as json_mod

    code = request.args.get("code")
    if not code:
        return "Missing authorization code", 400

    try:
        tokens = exchange_code(code)
        # Get the user's email from the access token
        resp = requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            timeout=30,
        )
        resp.raise_for_status()
        email = resp.json().get("email", "unknown")

        store_tokens(email, tokens)
        logger.info(f"Gmail OAuth completed for {email}")

        # Also store in env for persistence across restarts
        stored = os.environ.get("GMAIL_TOKENS", "{}")
        try:
            all_tokens = json_mod.loads(stored)
        except json_mod.JSONDecodeError:
            all_tokens = {}
        all_tokens[email] = tokens
        os.environ["GMAIL_TOKENS"] = json_mod.dumps(all_tokens)

        return f"Gmail connected for {email}! You can close this page and go back to WhatsApp.", 200
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return f"Error connecting Gmail: {str(e)}", 500


CRON_SECRET = os.environ.get("CRON_SECRET", "")


@app.route("/cron/leads", methods=["POST", "GET"])
def cron_leads():
    """5am daily automation: process MindBody lead emails and create draft replies.

    Secured with a secret token to prevent unauthorized access.
    Called by Railway cron job or external scheduler.
    """
    # Verify secret (skip if not configured — allows easy testing)
    token = request.args.get("token") or request.headers.get("X-Cron-Secret")
    if CRON_SECRET and token != CRON_SECRET:
        abort(403)

    try:
        from lead_automation import process_new_leads, format_lead_summary
        test_mode = request.args.get("test") == "1"
        drafts = process_new_leads(include_read=test_mode)
        summary = format_lead_summary(drafts)

        # Notify Sam on WhatsApp with the results
        if drafts:
            try:
                sam_whatsapp = "whatsapp:+61420233508"
                send_whatsapp_reply(sam_whatsapp, summary)
            except Exception as e:
                logger.error(f"Failed to send lead notification to Sam: {e}")

        logger.info(f"Lead automation complete: {len(drafts)} drafts created")
        return summary, 200

    except Exception as e:
        logger.error(f"Lead automation failed: {e}")
        return f"Error: {str(e)}", 500


@app.route("/cron/leads/debug", methods=["GET"])
def cron_leads_debug():
    """Debug endpoint: show what MindBody emails are in Erin's inbox."""
    token = request.args.get("token") or request.headers.get("X-Cron-Secret")
    if CRON_SECRET and token != CRON_SECRET:
        abort(403)

    from outlook_helper import read_inbox, read_email
    inbox = request.args.get("inbox", "admin@hiitaustralia.com.au")
    emails = read_inbox(count=20, search="mindbody", outlook_user=inbox)

    lines = [f"Inbox: {inbox}", f"Found: {len(emails)} emails matching 'mindbody'\n"]
    for e in emails:
        lines.append(f"From: {e['from_email']}")
        lines.append(f"Subject: {e['subject']}")
        lines.append(f"Read: {e['is_read']}")
        lines.append(f"Date: {e['date']}")
        lines.append(f"Preview: {e['preview'][:100]}")

        # Show full body for MindBody lead emails
        if "mindbodyemail" in e.get("from_email", "").lower() and "lead" in e.get("subject", "").lower():
            try:
                full = read_email(e["id"], outlook_user=inbox)
                lines.append(f"FULL BODY:\n{full['body'][:2000]}")
            except Exception as ex:
                lines.append(f"Error reading body: {ex}")
        lines.append("")

    return "\n".join(lines), 200


# ── Neuform Content Upload API ────────────────────────────────────────────────

@app.route("/api/drive/upload", methods=["POST", "OPTIONS"])
def drive_upload():
    """Upload a file to a specific Google Drive folder for the content calendar."""
    # CORS
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Upload-Key"
        resp.headers["Access-Control-Max-Age"] = "86400"
        return resp

    from flask import jsonify

    # Simple API key check
    upload_key = os.environ.get("NEUFORM_UPLOAD_KEY", "")
    provided_key = request.headers.get("X-Upload-Key", "") or request.form.get("key", "")
    if upload_key and provided_key != upload_key:
        return jsonify({"error": "Unauthorized"}), 401

    folder_id = request.form.get("folder_id")
    if not folder_id:
        return jsonify({"error": "Missing folder_id"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        import json as _json
        from google.oauth2 import service_account as _sa
        from googleapiclient.discovery import build as _build
        from googleapiclient.http import MediaIoBaseUpload

        sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        if sa_json:
            sa_info = _json.loads(sa_json)
            creds = _sa.Credentials.from_service_account_info(
                sa_info, scopes=["https://www.googleapis.com/auth/drive"]
            )
        else:
            return jsonify({"error": "No service account configured"}), 500

        drive = _build("drive", "v3", credentials=creds)

        media = MediaIoBaseUpload(file.stream, mimetype=file.content_type or "video/mp4", resumable=True)
        file_metadata = {"name": file.filename, "parents": [folder_id]}
        created = drive.files().create(body=file_metadata, media_body=media, fields="id,name,size").execute()

        resp = jsonify({"success": True, "fileId": created["id"], "name": created["name"]})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    except Exception as e:
        logger.error(f"Drive upload error: {e}")
        resp = jsonify({"error": str(e)})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 500


@app.route("/api/drive/folders", methods=["GET", "OPTIONS"])
def drive_folders():
    """Return the folder map for the content calendar."""
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    from flask import jsonify
    import json as _json
    from google.oauth2 import service_account as _sa
    from googleapiclient.discovery import build as _build

    try:
        sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        if not sa_json:
            return jsonify({"error": "No service account"}), 500

        sa_info = _json.loads(sa_json)
        creds = _sa.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        drive = _build("drive", "v3", credentials=creds)

        parent_id = "1m99uefGSWlDzT4XZvUk9zlLbb97fsK6T"
        folder_map = {}

        # Get day folders
        q = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        day_folders = drive.files().list(q=q, fields="files(id,name)", pageSize=100).execute().get("files", [])

        for day in sorted(day_folders, key=lambda x: x["name"]):
            day_data = {"id": day["id"], "posts": {}}

            # Get post folders
            q2 = f"'{day['id']}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            post_folders = drive.files().list(q=q2, fields="files(id,name)", pageSize=20).execute().get("files", [])

            for pf in sorted(post_folders, key=lambda x: x["name"]):
                post_num = pf["name"][:2]
                post_data = {"id": pf["id"], "name": pf["name"]}

                # Get subfolders (Raw/Edited/Approved)
                q3 = f"'{pf['id']}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                subs = drive.files().list(q=q3, fields="files(id,name)", pageSize=5).execute().get("files", [])
                for s in subs:
                    post_data[s["name"]] = s["id"]

                day_data["posts"][post_num] = post_data

            folder_map[day["name"]] = day_data

        resp = jsonify(folder_map)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Cache-Control"] = "public, max-age=3600"
        return resp

    except Exception as e:
        logger.error(f"Drive folders error: {e}")
        resp = jsonify({"error": str(e)})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return "OK", 200


@app.route("/", methods=["GET"])
def index():
    return "", 200


# ── Background scheduler: daily lead automation at 5am AEST ──────────────────

_scheduler_started = False


def _run_daily_leads():
    """Background thread that runs lead automation at 5am AEST every day."""
    from datetime import datetime, timezone, timedelta
    aest = timezone(timedelta(hours=10))

    logger.info("Lead automation scheduler started — will run daily at 5:00am AEST")

    while True:
        try:
            now = datetime.now(aest)
            # Calculate seconds until next 5:00am AEST
            target = now.replace(hour=5, minute=0, second=0, microsecond=0)
            if now >= target:
                # Already past 5am today — schedule for tomorrow
                target += timedelta(days=1)

            wait_seconds = (target - now).total_seconds()
            logger.info(
                f"Lead scheduler: next run at {target.strftime('%Y-%m-%d %H:%M AEST')} "
                f"({wait_seconds/3600:.1f} hours from now)"
            )
            time.sleep(wait_seconds)

            # Run lead automation
            logger.info("Lead scheduler: running daily lead automation...")
            from lead_automation import process_new_leads, format_lead_summary
            drafts = process_new_leads()
            summary = format_lead_summary(drafts)

            # Notify Sam on WhatsApp
            try:
                sam_whatsapp = "whatsapp:+61420233508"
                send_whatsapp_reply(sam_whatsapp, summary)
                logger.info(f"Lead scheduler: {len(drafts)} drafts created, Sam notified")
            except Exception as e:
                logger.error(f"Lead scheduler: failed to notify Sam: {e}")

        except Exception as e:
            logger.error(f"Lead scheduler error: {e}")
            # Sleep 60 seconds before retrying on error
            time.sleep(60)


def start_scheduler():
    """Start the background lead scheduler (only once, even with multiple gunicorn workers)."""
    global _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True

    t = threading.Thread(target=_run_daily_leads, daemon=True)
    t.start()
    logger.info("Background lead scheduler thread launched")


# Start scheduler when the app loads (gunicorn preload)
# Only start if we're actually running the server (not importing for tests)
if os.environ.get("PORT"):
    start_scheduler()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_scheduler()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
