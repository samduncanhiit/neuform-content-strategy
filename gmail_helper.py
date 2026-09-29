"""
Gmail integration via OAuth2 for reading drafts and emails.
Used for chontelhiit@gmail.com via Chonnie's WhatsApp number.
"""

import os
import json
import logging

import requests

logger = logging.getLogger(__name__)

GOOGLE_OAUTH_CLIENT_ID = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_OAUTH_CLIENT_SECRET = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
OAUTH_REDIRECT_URI = os.environ.get(
    "GOOGLE_OAUTH_REDIRECT_URI",
    "https://hiit-automations-production.up.railway.app/oauth/callback",
)

# Store refresh tokens in memory (persisted via env var for production)
_gmail_tokens = {}

GMAIL_API_BASE = "https://www.googleapis.com/gmail/v1"


def get_auth_url():
    """Get the OAuth2 authorization URL for Gmail access."""
    scopes = "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.compose"
    return (
        f"https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_OAUTH_CLIENT_ID}"
        f"&redirect_uri={OAUTH_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={scopes}"
        f"&access_type=offline"
        f"&prompt=consent"
    )


def exchange_code(code):
    """Exchange authorization code for tokens."""
    resp = requests.post("https://oauth2.googleapis.com/token", data={
        "code": code,
        "client_id": GOOGLE_OAUTH_CLIENT_ID,
        "client_secret": GOOGLE_OAUTH_CLIENT_SECRET,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "grant_type": "authorization_code",
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()


def refresh_access_token(refresh_token):
    """Get a new access token using a refresh token."""
    resp = requests.post("https://oauth2.googleapis.com/token", data={
        "refresh_token": refresh_token,
        "client_id": GOOGLE_OAUTH_CLIENT_ID,
        "client_secret": GOOGLE_OAUTH_CLIENT_SECRET,
        "grant_type": "refresh_token",
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()


def store_tokens(email, tokens):
    """Store tokens for a Gmail user."""
    _gmail_tokens[email] = tokens
    logger.info(f"Gmail tokens stored for {email}")


def get_access_token(email):
    """Get a valid access token for a Gmail user."""
    tokens = _gmail_tokens.get(email)
    if not tokens:
        # Try loading from env var
        stored = os.environ.get("GMAIL_TOKENS")
        if stored:
            try:
                all_tokens = json.loads(stored)
                if email in all_tokens:
                    _gmail_tokens[email] = all_tokens[email]
                    tokens = all_tokens[email]
            except (json.JSONDecodeError, KeyError):
                pass

    if not tokens:
        return None

    # Refresh the access token
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        return None

    try:
        new_tokens = refresh_access_token(refresh_token)
        tokens["access_token"] = new_tokens["access_token"]
        _gmail_tokens[email] = tokens
        return new_tokens["access_token"]
    except Exception as e:
        logger.error(f"Failed to refresh Gmail token for {email}: {e}")
        return None


def _auth_headers(email):
    token = get_access_token(email)
    if not token:
        raise RuntimeError(f"Gmail not connected for {email}. Send 'connect gmail' to set up.")
    return {"Authorization": f"Bearer {token}"}


def _extract_headers(msg):
    """Extract a header name→value dict from a Gmail message payload."""
    return {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}


def read_gmail_inbox(email, count=10, query=None):
    """Read recent emails from Gmail inbox."""
    params = {"maxResults": min(count, 20)}
    if query:
        params["q"] = query
    else:
        params["q"] = "in:inbox"

    resp = requests.get(
        f"{GMAIL_API_BASE}/users/me/messages",
        headers=_auth_headers(email),
        params=params,
        timeout=30,
    )
    resp.raise_for_status()

    messages = resp.json().get("messages", [])
    results = []

    for msg_ref in messages[:count]:
        msg_resp = requests.get(
            f"{GMAIL_API_BASE}/users/me/messages/{msg_ref['id']}",
            headers=_auth_headers(email),
            params={"format": "metadata", "metadataHeaders": ["Subject", "From", "Date"]},
            timeout=30,
        )
        msg_resp.raise_for_status()
        msg = msg_resp.json()
        hdrs = _extract_headers(msg)

        results.append({
            "id": msg.get("id"),
            "subject": hdrs.get("Subject", "(no subject)"),
            "from": hdrs.get("From", ""),
            "date": hdrs.get("Date", ""),
            "snippet": msg.get("snippet", ""),
            "is_read": "UNREAD" not in msg.get("labelIds", []),
        })

    return results


def read_gmail_drafts(email, count=10):
    """Read drafts from Gmail."""
    resp = requests.get(
        f"{GMAIL_API_BASE}/users/me/drafts",
        headers=_auth_headers(email),
        params={"maxResults": min(count, 20)},
        timeout=30,
    )
    resp.raise_for_status()

    drafts = resp.json().get("drafts", [])
    results = []

    for draft_ref in drafts[:count]:
        draft_resp = requests.get(
            f"{GMAIL_API_BASE}/users/me/drafts/{draft_ref['id']}",
            headers=_auth_headers(email),
            params={"format": "metadata"},
            timeout=30,
        )
        draft_resp.raise_for_status()
        draft = draft_resp.json()
        msg = draft.get("message", {})
        hdrs = _extract_headers(msg)

        results.append({
            "id": draft_ref.get("id"),
            "subject": hdrs.get("Subject", "(no subject)"),
            "to": hdrs.get("To", ""),
            "snippet": msg.get("snippet", ""),
        })

    return results


def format_gmail_inbox(emails):
    """Format Gmail inbox for WhatsApp."""
    if not emails:
        return "Gmail inbox is empty."

    lines = [f"*Gmail Inbox* ({len(emails)} recent)\n"]
    for i, e in enumerate(emails, 1):
        read_marker = "" if e["is_read"] else "🔵 "
        lines.append(f"{read_marker}{i}. *{e['subject']}*\n   From: {e['from']}\n   {e['snippet'][:80]}")
    return "\n\n".join(lines)


def format_gmail_drafts(drafts):
    """Format Gmail drafts for WhatsApp."""
    if not drafts:
        return "No drafts in Gmail."

    lines = [f"*Gmail Drafts* ({len(drafts)})\n"]
    for i, d in enumerate(drafts, 1):
        to = f"\n   To: {d['to']}" if d.get("to") else ""
        lines.append(f"{i}. *{d['subject']}*{to}\n   {d['snippet'][:80]}")
    return "\n\n".join(lines)
