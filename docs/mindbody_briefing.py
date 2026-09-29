"""
MindBody Daily Briefing Script
Connects to MindBody Public API V6.0 sandbox and generates a daily summary.
"""

import os
import sys
from datetime import datetime, timedelta

import requests

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_URL = "https://api.mindbodyonline.com/public/v6"

API_KEY = os.environ.get("MINDBODY_API_KEY")
SITE_ID = os.environ.get("MINDBODY_SITE_ID", "-99")
SOURCE_NAME = os.environ.get("MINDBODY_SOURCE_NAME")
SOURCE_PASSWORD = os.environ.get("MINDBODY_SOURCE_PASSWORD")

REQUIRED_VARS = {
    "MINDBODY_API_KEY": API_KEY,
    "MINDBODY_SOURCE_NAME": SOURCE_NAME,
    "MINDBODY_SOURCE_PASSWORD": SOURCE_PASSWORD,
}


def check_env():
    missing = [k for k, v in REQUIRED_VARS.items() if not v]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)


# ── HTTP helpers ───────────────────────────────────────────────────────────────


def base_headers():
    return {
        "Content-Type": "application/json",
        "Api-Key": API_KEY,
        "SiteId": SITE_ID,
    }


def authenticate():
    """Obtain a staff user token using source credentials."""
    url = f"{BASE_URL}/usertoken/issue"
    payload = {
        "Username": f"_{SOURCE_NAME}",
        "Password": SOURCE_PASSWORD,
    }
    resp = requests.post(url, json=payload, headers=base_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    token = data.get("AccessToken")
    if not token:
        print("ERROR: Authentication failed — no AccessToken returned.")
        print(f"Response: {data}")
        sys.exit(1)
    return token


def auth_headers(token):
    h = base_headers()
    h["authorization"] = token
    return h


def api_get(path, token, params=None):
    """GET request with automatic pagination (up to 200 per page)."""
    url = f"{BASE_URL}/{path}"
    resp = requests.get(url, headers=auth_headers(token), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Data fetchers ──────────────────────────────────────────────────────────────


def get_classes_today(token):
    """Fetch today's class schedule."""
    today = datetime.now().strftime("%Y-%m-%dT00:00:00")
    end = datetime.now().strftime("%Y-%m-%dT23:59:59")
    params = {
        "startDateTime": today,
        "endDateTime": end,
        "hideCanceledClasses": "true",
        "limit": 200,
        "offset": 0,
    }
    data = api_get("class/classes", token, params)
    return data.get("Classes", [])


def get_class_visits(token, class_id):
    """Fetch bookings/visits for a specific class."""
    params = {"classID": class_id}
    data = api_get("class/classvisits", token, params)
    return data.get("Visits", [])


def get_active_clients(token, max_pages=5):
    """Fetch active clients (paginated, capped to avoid rate limits)."""
    all_clients = []
    offset = 0
    limit = 200
    for _ in range(max_pages):
        try:
            params = {"limit": limit, "offset": offset}
            data = api_get("client/clients", token, params)
            clients = data.get("Clients", [])
            all_clients.extend(clients)
            if len(clients) < limit:
                break
            offset += limit
        except requests.HTTPError:
            break
    return all_clients


def get_payment_failures(token):
    """Check for failed transactions today."""
    today = datetime.now().strftime("%Y-%m-%dT00:00:00")
    end = datetime.now().strftime("%Y-%m-%dT23:59:59")
    params = {
        "TransactionStartDateTime": today,
        "TransactionEndDateTime": end,
        "limit": 200,
        "offset": 0,
    }
    data = api_get("sale/transactions", token, params)
    transactions = data.get("Transactions", [])
    # Filter for failed / declined statuses
    failed = [
        t for t in transactions
        if str(t.get("Status", "")).lower() in ("failed", "declined", "error")
    ]
    return failed, transactions


# ── Formatting ─────────────────────────────────────────────────────────────────


def fmt_time(iso_str):
    """Parse ISO datetime string and return a human-readable time."""
    if not iso_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%-I:%M %p")
    except (ValueError, TypeError):
        return iso_str


def print_briefing(classes, class_bookings, clients, failed_txns, all_txns):
    today_str = datetime.now().strftime("%A, %B %-d, %Y")

    print("=" * 64)
    print(f"  HIIT Station Capalaba — Daily Briefing")
    print(f"  {today_str}")
    print("=" * 64)

    # ── Class Schedule ─────────────────────────────────────────────
    print(f"\n{'─' * 64}")
    print("  CLASS SCHEDULE")
    print(f"{'─' * 64}")

    if not classes:
        print("  No classes scheduled today.")
    else:
        for cls in classes:
            name = cls.get("ClassDescription", {}).get("Name", "Unknown Class")
            start = fmt_time(cls.get("StartDateTime"))
            end = fmt_time(cls.get("EndDateTime"))
            staff = cls.get("Staff", {})
            instructor = f"{staff.get('FirstName', '')} {staff.get('LastName', '')}".strip() or "TBA"
            max_cap = cls.get("MaxCapacity", "?")
            booked = cls.get("TotalBooked", 0)
            waitlisted = cls.get("TotalBookedWaitlist", 0)
            class_id = cls.get("Id")
            visit_count = class_bookings.get(class_id, booked)

            print(f"\n  {name}")
            print(f"    Time:       {start} – {end}")
            print(f"    Instructor: {instructor}")
            print(f"    Booked:     {visit_count}/{max_cap}", end="")
            if waitlisted:
                print(f"  (+{waitlisted} waitlisted)", end="")
            print()

    total_bookings = sum(
        cls.get("TotalBooked", 0) for cls in classes
    )
    print(f"\n  Total classes: {len(classes)}  |  Total bookings: {total_bookings}")

    # ── Active Clients ─────────────────────────────────────────────
    print(f"\n{'─' * 64}")
    print("  ACTIVE CLIENTS / MEMBERS")
    print(f"{'─' * 64}")

    if not clients:
        print("  No active clients found.")
    else:
        print(f"  Total active clients: {len(clients)}\n")
        # Show first 20 as a summary
        for i, c in enumerate(clients[:20], 1):
            first = c.get("FirstName") or ""
            last = c.get("LastName") or ""
            email = c.get("Email") or "N/A"
            status = c.get("Status") or ""
            print(f"  {i:>3}. {first} {last:<20} {email:<30} [{status}]")
        if len(clients) > 20:
            print(f"\n  ... and {len(clients) - 20} more active clients")

    # ── Payment Failures ───────────────────────────────────────────
    print(f"\n{'─' * 64}")
    print("  PAYMENT STATUS")
    print(f"{'─' * 64}")

    print(f"  Transactions today: {len(all_txns)}")
    if not failed_txns:
        print("  Payment failures:   None ✓")
    else:
        print(f"  Payment failures:   {len(failed_txns)}")
        print()
        for t in failed_txns:
            client_id = t.get("ClientId", "?")
            amount = t.get("Amount", 0)
            status = t.get("Status", "Unknown")
            txn_id = t.get("Id", "?")
            print(f"    • Transaction #{txn_id} — Client {client_id} — ${amount:.2f} — {status}")

    print(f"\n{'=' * 64}")
    print("  End of briefing")
    print(f"{'=' * 64}\n")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    check_env()

    print("Authenticating with MindBody API...")
    token = authenticate()
    print("Authenticated successfully.\n")

    print("Fetching today's class schedule...")
    classes = get_classes_today(token)

    print("Fetching bookings per class...")
    class_bookings = {}
    for cls in classes:
        class_id = cls.get("Id")
        if class_id:
            try:
                visits = get_class_visits(token, class_id)
                class_bookings[class_id] = len(visits)
            except requests.HTTPError:
                # Fall back to TotalBooked from the class data
                class_bookings[class_id] = cls.get("TotalBooked", 0)

    print("Fetching active clients...")
    clients = get_active_clients(token)

    print("Checking payment status...")
    failed_txns, all_txns = get_payment_failures(token)

    print()
    print_briefing(classes, class_bookings, clients, failed_txns, all_txns)


if __name__ == "__main__":
    main()
