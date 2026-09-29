"""
Outlook integration via Microsoft Graph API.
Provides email reading and drafting for sam@hiitaustralia.com.au.
"""

import os
import logging

import msal
import requests

logger = logging.getLogger(__name__)

MS_CLIENT_ID = os.environ.get("MS_CLIENT_ID")
MS_CLIENT_SECRET = os.environ.get("MS_CLIENT_SECRET")
MS_TENANT_ID = os.environ.get("MS_TENANT_ID")
OUTLOOK_USER = os.environ.get("OUTLOOK_USER", "sam@hiitaustralia.com.au")

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

_ms_app = None


def _get_ms_app():
    global _ms_app
    if _ms_app is None:
        _ms_app = msal.ConfidentialClientApplication(
            MS_CLIENT_ID,
            authority=f"https://login.microsoftonline.com/{MS_TENANT_ID}",
            client_credential=MS_CLIENT_SECRET,
        )
    return _ms_app


def _get_token():
    app = _get_ms_app()
    result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    if "access_token" not in result:
        raise RuntimeError(f"Failed to get Graph token: {result.get('error_description', result)}")
    return result["access_token"]


def _headers():
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Content-Type": "application/json",
    }


def read_inbox(count=10, search=None, outlook_user=None):
    """Read recent emails from inbox.

    Args:
        count: Number of emails to fetch (max 50).
        search: Optional search query (searches subject, body, sender).
        outlook_user: Email address to read from (defaults to OUTLOOK_USER env var).

    Returns:
        List of dicts with subject, from, date, preview, id.
    """
    user = outlook_user or OUTLOOK_USER
    params = {
        "$top": min(count, 50),
        "$select": "id,subject,from,receivedDateTime,bodyPreview,isRead",
        "$orderby": "receivedDateTime desc",
    }
    if search:
        params["$search"] = f'"{search}"'
        params.pop("$orderby")  # $orderby not supported with $search

    url = f"{GRAPH_BASE}/users/{user}/messages"
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    resp.raise_for_status()

    emails = []
    for m in resp.json().get("value", []):
        sender = m.get("from", {}).get("emailAddress", {})
        emails.append({
            "id": m.get("id"),
            "subject": m.get("subject", "(no subject)"),
            "from_name": sender.get("name", ""),
            "from_email": sender.get("address", ""),
            "date": m.get("receivedDateTime", ""),
            "preview": m.get("bodyPreview", "")[:200],
            "is_read": m.get("isRead", False),
        })
    return emails


def read_email(message_id, outlook_user=None):
    """Read full content of a specific email.

    Args:
        message_id: The Graph API message ID.
        outlook_user: Email address to read from (defaults to OUTLOOK_USER env var).

    Returns:
        Dict with full email details including body.
    """
    user = outlook_user or OUTLOOK_USER
    url = f"{GRAPH_BASE}/users/{user}/messages/{message_id}"
    params = {"$select": "id,subject,from,toRecipients,ccRecipients,receivedDateTime,body,bodyPreview"}
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    resp.raise_for_status()
    m = resp.json()

    sender = m.get("from", {}).get("emailAddress", {})
    to_list = [r.get("emailAddress", {}).get("address", "") for r in m.get("toRecipients", [])]
    cc_list = [r.get("emailAddress", {}).get("address", "") for r in m.get("ccRecipients", [])]

    return {
        "id": m.get("id"),
        "subject": m.get("subject", "(no subject)"),
        "from_name": sender.get("name", ""),
        "from_email": sender.get("address", ""),
        "to": to_list,
        "cc": cc_list,
        "date": m.get("receivedDateTime", ""),
        "body": m.get("body", {}).get("content", ""),
        "body_type": m.get("body", {}).get("contentType", "text"),
    }


def _parse_recipients(addresses):
    """Parse a comma-separated string of email addresses into Graph API format."""
    return [
        {"emailAddress": {"address": addr.strip()}}
        for addr in addresses.split(",") if addr.strip()
    ]


def create_draft(to, subject, body, cc=None, reply_to_id=None, outlook_user=None, body_type="text"):
    """Create an email draft.

    Args:
        to: Recipient email address (or comma-separated list).
        subject: Email subject.
        body: Email body (plain text or HTML).
        cc: Optional CC email address(es).
        reply_to_id: Optional message ID to reply to.
        outlook_user: Email address to create draft from (defaults to OUTLOOK_USER env var).
        body_type: "text" for plain text, "html" for HTML content.

    Returns:
        Dict with draft id and webLink.
    """
    user = outlook_user or OUTLOOK_USER

    message = {
        "subject": subject,
        "body": {"contentType": body_type, "content": body},
        "toRecipients": _parse_recipients(to),
    }
    if cc:
        message["ccRecipients"] = _parse_recipients(cc)

    if reply_to_id:
        url = f"{GRAPH_BASE}/users/{user}/messages/{reply_to_id}/createReply"
        resp = requests.post(url, headers=_headers(), json={"comment": body}, timeout=30)
        resp.raise_for_status()
    else:
        url = f"{GRAPH_BASE}/users/{user}/messages"
        resp = requests.post(url, headers=_headers(), json={"message": message, "isDraft": True}, timeout=30)
        if resp.status_code == 404 or resp.status_code == 400:
            # Try alternate draft endpoint
            resp = requests.post(
                f"{GRAPH_BASE}/users/{user}/messages",
                headers=_headers(),
                json=message,
                timeout=30,
            )
        resp.raise_for_status()

    draft = resp.json()
    return {
        "id": draft.get("id"),
        "subject": draft.get("subject"),
        "web_link": draft.get("webLink", ""),
    }


def send_email(to, subject, body, cc=None):
    """Send an email directly (not as draft).

    Args:
        to: Recipient email address (or comma-separated list).
        subject: Email subject.
        body: Email body (plain text).
        cc: Optional CC email address(es).
    """
    message = {
        "subject": subject,
        "body": {"contentType": "text", "content": body},
        "toRecipients": _parse_recipients(to),
    }
    if cc:
        message["ccRecipients"] = _parse_recipients(cc)

    url = f"{GRAPH_BASE}/users/{OUTLOOK_USER}/sendMail"
    resp = requests.post(url, headers=_headers(), json={"message": message}, timeout=30)
    resp.raise_for_status()


def format_inbox_summary(emails):
    """Format a list of emails into a WhatsApp-friendly summary."""
    if not emails:
        return "Your inbox is empty."

    lines = [f"📧 *Inbox* ({len(emails)} recent emails):\n"]
    for i, e in enumerate(emails, 1):
        read_marker = "" if e["is_read"] else "🔵 "
        date_short = e["date"][:10] if e["date"] else ""
        lines.append(
            f"{read_marker}{i}. *{e['subject']}*\n"
            f"   From: {e['from_name'] or e['from_email']}\n"
            f"   {date_short}"
        )
    return "\n".join(lines)
