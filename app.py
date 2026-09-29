"""
WhatsApp Bot for HIIT Station Capalaba
Receives messages via Twilio webhook, processes with Claude API, responds via WhatsApp.
"""

import os
import sys
import logging
import traceback
import threading
import time
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

# ── Conversation history (per-sender, in-memory, TTL-bounded) ────────────────

_history_lock = threading.Lock()
_user_history = {}  # sender -> {"messages": [...], "last_ts": float}
HISTORY_TTL_SEC = 15 * 60  # drop history after 15 min of inactivity
HISTORY_MAX_MESSAGES = 12  # ≈6 user + 6 assistant turns


def _load_history(sender):
    """Return a copy of the sender's recent text-only conversation, or []."""
    if not sender:
        return []
    now = time.time()
    with _history_lock:
        entry = _user_history.get(sender)
        if not entry:
            return []
        if now - entry["last_ts"] > HISTORY_TTL_SEC:
            del _user_history[sender]
            return []
        return list(entry["messages"])


def _save_history(sender, messages):
    """Trim and persist conversation history for a sender."""
    if not sender:
        return
    trimmed = messages[-HISTORY_MAX_MESSAGES:]
    # Anthropic requires the first message to be role=user
    while trimmed and trimmed[0].get("role") != "user":
        trimmed.pop(0)
    with _history_lock:
        _user_history[sender] = {"messages": trimmed, "last_ts": time.time()}


def _reset_history(sender):
    if not sender:
        return
    with _history_lock:
        _user_history.pop(sender, None)


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

_MINDBODY_TOOLS = [
    {
        "name": "get_todays_classes",
        "description": "Get today's class schedule with names, times, instructors, bookings",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_daily_briefing",
        "description": "Daily briefing: today's classes and bookings",
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
        "name": "run_class_report",
        "description": "New client report on a class: first-timers, intro/trial pricing, new memberships (14d). Can run on a single class or ALL classes for a day.",
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string", "description": "Class name e.g. 'HIIT Rox', 'HIIT Maxx'. Omit to run report on ALL classes for the day."},
                "class_date": {"type": "string", "description": "YYYY-MM-DD (default today)"},
                "class_time": {"type": "string", "description": "Time to pick a specific class, e.g. '6am', '18:00', '6:00 PM'. If given without class_name, returns the one class at that time."},
            },
            "required": [],
        },
    },
    {
        "name": "get_noshow_report",
        "description": "No-show report: clients booked but not signed in after class finished. Use when user asks 'who didn't show up', 'no shows', 'didn't sign in'. Can run on a single class or ALL classes for a day.",
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string", "description": "Class name e.g. 'HIIT Rox', 'HIIT Maxx'. Omit to run report on ALL classes for the day."},
                "class_date": {"type": "string", "description": "YYYY-MM-DD (default today)"},
                "class_time": {"type": "string", "description": "Time to pick a specific class, e.g. '6am', '18:00', '6:00 PM'. If given without class_name, returns the one class at that time."},
            },
            "required": [],
        },
    },
]

def handle_tool_call(tool_name, tool_input):
    """Execute a tool call and return the result."""
    if tool_name == "get_todays_classes":
        from mindbody_helper import get_todays_schedule, format_schedule
        classes = get_todays_schedule()
        return format_schedule(classes)

    elif tool_name == "get_daily_briefing":
        from mindbody_helper import get_daily_briefing, format_briefing
        return format_briefing(get_daily_briefing())

    elif tool_name == "search_clients":
        from mindbody_helper import search_clients, format_clients
        search_text = tool_input.get("search_text", "")[:100]  # Cap search length
        clients = search_clients(search_text)
        return format_clients(clients)

    elif tool_name == "get_client_detail":
        from mindbody_helper import get_client_detail, format_client_detail
        name = (tool_input.get("client_name") or "")[:100]
        days = tool_input.get("days_back")
        if days is not None:
            days = max(1, min(int(days or 0), 365))
        result = get_client_detail(client_name=name, days_back=days)
        return format_client_detail(result)

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

    elif tool_name == "get_arrears_report":
        from mindbody_helper import get_arrears_report, format_arrears_report
        report = get_arrears_report()
        return format_arrears_report(report)

    elif tool_name == "get_weekly_summary":
        from mindbody_helper import get_weekly_summary, format_weekly_summary
        summary = get_weekly_summary()
        return format_weekly_summary(summary)

    elif tool_name == "run_class_report":
        from mindbody_helper import run_class_report, run_multi_class_report, format_class_report, format_multi_class_report
        class_name = tool_input.get("class_name")
        class_date = tool_input.get("class_date")
        class_time = tool_input.get("class_time")
        if class_name:
            report = run_class_report(class_name=class_name, class_date=class_date, class_time=class_time)
            return format_class_report(report)
        else:
            reports = run_multi_class_report(class_date=class_date, class_time=class_time)
            return format_multi_class_report(reports)

    elif tool_name == "get_noshow_report":
        from mindbody_helper import get_noshow_report, get_multi_noshow_report, format_noshow_report, format_multi_noshow_report
        class_name = tool_input.get("class_name")
        class_date = tool_input.get("class_date")
        class_time = tool_input.get("class_time")
        if class_name:
            report = get_noshow_report(class_name=class_name, class_date=class_date, class_time=class_time)
            return format_noshow_report(report)
        else:
            reports = get_multi_noshow_report(class_date=class_date, class_time=class_time)
            return format_multi_noshow_report(reports)

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

    parts.append(f"Today is {today_str}.")
    parts.append(
        "When user says 'next week' they mean the upcoming Mon-Sun. "
        "Calculate correct dates from today."
    )

    return "\n\n".join(parts)


def get_claude_response(user_message, sender=None):
    """Send a message to Claude with tools and return the text response."""
    client = get_claude_client()

    raw_number = (sender or "").replace("whatsapp:", "").strip()

    if user_message.strip().lower() in ("/reset", "reset chat", "new chat", "clear chat"):
        _reset_history(raw_number)
        return "Chat history cleared. What can I help you with?"

    history = _load_history(raw_number)
    messages = history + [{"role": "user", "content": user_message}]
    logger.info(f"History: loaded {len(history)} prior messages for {mask_number(raw_number)}")

    user_name = USER_NAMES.get(raw_number)
    system = _build_system_prompt(user_name, raw_number)
    tools = _MINDBODY_TOOLS

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=system,
        tools=tools,
        messages=messages,
    )
    logger.info(f"Claude first response: stop_reason={response.stop_reason} out_tokens={response.usage.output_tokens}")

    # Detect if the user is asking about no-shows so we can correct wrong tool usage
    _noshow_keywords = ("no show", "no-show", "noshow", "didn't show", "didn't sign in",
                        "not signed in", "didn't attend", "didn't turn up", "who missed")
    user_wants_noshow = any(kw in user_message.lower() for kw in _noshow_keywords)

    # Handle tool use loop (max 8 iterations to prevent runaway loops)
    tool_iterations = 0
    while response.stop_reason == "tool_use" and tool_iterations < 8:
        tool_iterations += 1
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for block in assistant_content:
            if block.type == "tool_use":
                logger.info(f"Tool call: {block.name}")
                try:
                    result = handle_tool_call(block.name, block.input)
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

        total_result_chars = sum(len(r.get("content", "")) for r in tool_results)
        logger.info(f"Tool loop iteration {tool_iterations}: {len(tool_results)} results, {total_result_chars} chars total")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=system,
            tools=tools,
            messages=messages,
        )
        logger.info(f"Claude loop response: stop_reason={response.stop_reason} out_tokens={response.usage.output_tokens}")

    final_text = None
    for block in response.content:
        if getattr(block, "type", None) == "text" and block.text:
            final_text = block.text
            break

    if final_text is None:
        logger.warning(f"No text in final response. stop_reason={response.stop_reason} blocks={[getattr(b,'type',None) for b in response.content]}")
        if response.stop_reason == "max_tokens":
            final_text = "Sorry, that request was too big for me to finish in one go. Try splitting it into smaller chunks."
        else:
            final_text = "I processed your request but have no response to show."

    # Persist clean history: prior turns + this user message + the assistant's text reply.
    # We intentionally drop tool_use/tool_result blocks — they bloat context and aren't
    # needed for follow-up conversation (fresh data should be re-fetched via tools).
    _save_history(raw_number, history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": final_text},
    ])
    return final_text


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


WHATSAPP_CHAR_LIMIT = 1500  # Twilio WhatsApp rejects >1600 (error 21617); leave buffer


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
        logger.error(f"Claude API error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
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


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return "OK", 200


@app.route("/", methods=["GET"])
def index():
    return "", 200


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
