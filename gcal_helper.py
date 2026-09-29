"""
Google Calendar integration via Service Account.
"""

import os
import json
import logging
from datetime import datetime, timedelta

from google.oauth2 import service_account
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]

_service = None


def _get_service():
    global _service
    if _service is not None:
        return _service

    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not set")

    creds_info = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    _service = build("calendar", "v3", credentials=creds)
    return _service


def _get_calendar_id():
    """Get the calendar ID — uses env var or defaults to primary."""
    return os.environ.get("GOOGLE_CALENDAR_ID", "primary")


def get_events(days_forward=1, days_back=0, calendar_id=None):
    """Get calendar events for a date range.

    Args:
        days_forward: Days into the future (default 1 = today only).
        days_back: Days into the past (default 0).
        calendar_id: Optional calendar ID override.
    """
    service = _get_service()
    cal_id = calendar_id or _get_calendar_id()

    start = (datetime.utcnow() - timedelta(days=days_back)).replace(hour=0, minute=0, second=0)
    end = (datetime.utcnow() + timedelta(days=days_forward)).replace(hour=23, minute=59, second=59)

    time_min = start.isoformat() + "Z"
    time_max = end.isoformat() + "Z"

    results = service.events().list(
        calendarId=cal_id,
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy="startTime",
        maxResults=50,
    ).execute()

    events = results.get("items", [])

    parsed = []
    for event in events:
        start_raw = event.get("start", {})
        end_raw = event.get("end", {})

        # All-day events use "date", timed events use "dateTime"
        if start_raw.get("dateTime"):
            try:
                start_dt = datetime.fromisoformat(start_raw["dateTime"].replace("Z", "+00:00"))
                start_fmt = start_dt.strftime("%-I:%M %p")
                date_str = start_dt.strftime("%a %-d %b")
            except (ValueError, TypeError):
                start_fmt = start_raw["dateTime"]
                date_str = ""
        else:
            start_fmt = "All day"
            date_str = start_raw.get("date", "")

        if end_raw.get("dateTime"):
            try:
                end_dt = datetime.fromisoformat(end_raw["dateTime"].replace("Z", "+00:00"))
                end_fmt = end_dt.strftime("%-I:%M %p")
            except (ValueError, TypeError):
                end_fmt = end_raw["dateTime"]
        else:
            end_fmt = ""

        parsed.append({
            "summary": event.get("summary", "(No title)"),
            "date": date_str,
            "start": start_fmt,
            "end": end_fmt,
            "location": event.get("location", ""),
            "description": (event.get("description") or "")[:100],
        })

    return parsed


def get_todays_events():
    return get_events(days_forward=0, days_back=0)


def create_event(summary, start_date, start_time, end_time, description=None, location=None, calendar_id=None):
    """Create a calendar event.

    Args:
        summary: Event title.
        start_date: Date string YYYY-MM-DD.
        start_time: Start time HH:MM (24hr).
        end_time: End time HH:MM (24hr).
        description: Optional description.
        location: Optional location.
        calendar_id: Optional calendar ID override.
    """
    service = _get_service()
    cal_id = calendar_id or _get_calendar_id()

    timezone = "Australia/Brisbane"

    event_body = {
        "summary": summary,
        "start": {
            "dateTime": f"{start_date}T{start_time}:00",
            "timeZone": timezone,
        },
        "end": {
            "dateTime": f"{start_date}T{end_time}:00",
            "timeZone": timezone,
        },
    }

    if description:
        event_body["description"] = description
    if location:
        event_body["location"] = location

    event = service.events().insert(calendarId=cal_id, body=event_body).execute()

    return {
        "id": event.get("id"),
        "summary": event.get("summary"),
        "link": event.get("htmlLink", ""),
    }


def create_allday_event(summary, date, description=None):
    """Create an all-day calendar event.

    Args:
        summary: Event title.
        date: Date string YYYY-MM-DD.
        description: Optional description.
    """
    service = _get_service()
    cal_id = _get_calendar_id()

    event_body = {
        "summary": summary,
        "start": {"date": date},
        "end": {"date": date},
    }

    if description:
        event_body["description"] = description

    event = service.events().insert(calendarId=cal_id, body=event_body).execute()

    return {
        "id": event.get("id"),
        "summary": event.get("summary"),
        "link": event.get("htmlLink", ""),
    }


def format_events(events, title="Calendar"):
    """Format events for WhatsApp."""
    if not events:
        return f"*{title}*\nNo events scheduled."

    lines = [f"*{title}*\n"]
    for e in events:
        time_str = e["start"]
        if e["end"] and e["start"] != "All day":
            time_str = f"{e['start']} - {e['end']}"

        date_part = f"{e['date']} | " if e.get("date") else ""
        location = f"\n  Location: {e['location']}" if e.get("location") else ""

        lines.append(f"*{e['summary']}*\n  {date_part}{time_str}{location}")

    return "\n\n".join(lines)
