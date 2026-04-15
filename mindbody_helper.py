"""
MindBody API helper for WhatsApp bot integration.
Provides class schedule, bookings, client stats, and payment status.
"""

import os
import logging
import threading
from datetime import datetime, timedelta, timezone

import requests

# AEST timezone (UTC+10)
AEST = timezone(timedelta(hours=10))

logger = logging.getLogger(__name__)


def _now():
    """Get current time in AEST."""
    return datetime.now(AEST)

BASE_URL = "https://api.mindbodyonline.com/public/v6"

API_KEY = os.environ.get("MINDBODY_API_KEY")
SITE_ID = os.environ.get("MINDBODY_SITE_ID", "-99")
SOURCE_NAME = os.environ.get("MINDBODY_SOURCE_NAME")
SOURCE_PASSWORD = os.environ.get("MINDBODY_SOURCE_PASSWORD")

_mb_token = None
_mb_token_time = None
_mb_token_lock = threading.Lock()

# ── Cache ─────────────────────────────────────────────────────────────────────
# Caches expensive API lookups for 30 minutes to reduce API calls

CACHE_TTL_MEMBERS = 518400  # 6 days for membership data
CACHE_TTL_CLASSES = 3600     # 1 hour for class schedules

_cache = {}
_cache_lock = threading.Lock()


def _cache_get(key, ttl=None):
    """Get a value from cache if it exists and hasn't expired."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        value, timestamp = entry
        max_age = ttl or CACHE_TTL_MEMBERS
        if (datetime.now() - timestamp).total_seconds() > max_age:
            del _cache[key]
            return None
        return value


def _cache_set(key, value):
    """Store a value in cache."""
    with _cache_lock:
        _cache[key] = (value, datetime.now())


def _cache_clear():
    """Clear all cached data."""
    with _cache_lock:
        _cache.clear()


def _client_name(client):
    """Extract full name from a MindBody client dict."""
    return f"{client.get('FirstName') or ''} {client.get('LastName') or ''}".strip() or "Unknown"


def _base_headers():
    return {
        "Content-Type": "application/json",
        "Api-Key": API_KEY,
        "SiteId": SITE_ID,
    }


def _get_token():
    """Get or refresh MindBody auth token (thread-safe)."""
    global _mb_token, _mb_token_time
    with _mb_token_lock:
        if _mb_token and _mb_token_time:
            elapsed = (datetime.now() - _mb_token_time).total_seconds()
            if elapsed < 600:
                return _mb_token

        url = f"{BASE_URL}/usertoken/issue"
        payload = {
            "Username": f"_{SOURCE_NAME}",
            "Password": SOURCE_PASSWORD,
        }
        resp = requests.post(url, json=payload, headers=_base_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        _mb_token = data.get("AccessToken")
        _mb_token_time = datetime.now()
        if not _mb_token:
            raise RuntimeError("MindBody auth failed")
        return _mb_token


def _auth_headers():
    h = _base_headers()
    h["authorization"] = _get_token()
    return h


def _api_get(path, params=None):
    url = f"{BASE_URL}/{path}"
    resp = requests.get(url, headers=_auth_headers(), params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _paginated_get(path, result_key, params=None, max_pages=50):
    """Generic paginated GET — fetches all pages of a MindBody endpoint.

    Args:
        path: API path (e.g. "client/clients").
        result_key: JSON key containing the list (e.g. "Clients", "Transactions").
        params: Extra query params (merged with limit/offset).
        max_pages: Safety cap on number of pages.
    """
    all_items = []
    offset = 0
    base_params = dict(params) if isinstance(params, dict) else {}
    # Handle list-of-tuples params (for repeated keys like ClientIds)
    tuple_params = params if isinstance(params, list) else []

    for _ in range(max_pages):
        try:
            p = {**base_params, "limit": 200, "offset": offset}
            if tuple_params:
                p = list(tuple_params) + list(p.items())
            data = _api_get(path, p)
            items = data.get(result_key, [])
            all_items.extend(items)
            if len(items) < 200:
                break
            offset += 200
        except requests.HTTPError:
            break
    return all_items


def _get_all_clients_paginated(params, max_pages=10):
    """Fetch clients with pagination."""
    return _paginated_get("client/clients", "Clients", params, max_pages)


# ── Cached lookups ────────────────────────────────────────────────────────────


def _get_tracked_members():
    """Get all clients with tracked memberships using bulk API. Cached for 6 days.

    Returns dict: {client_id: {"name": str, "membership": str, "status": str}}
    """
    cached = _cache_get("tracked_members")
    if cached is not None:
        logger.info("Using cached tracked members data")
        return cached

    logger.info("Fetching tracked members via bulk API (not cached)")
    all_clients = _get_all_clients_paginated({})

    # Build a lookup of client info by ID
    client_info = {}
    client_ids = []
    for c in all_clients:
        cid = c.get("Id")
        if cid:
            client_info[str(cid)] = {
                "name": _client_name(c),
                "status": c.get("Status") or "Unknown",
            }
            client_ids.append(str(cid))

    # Bulk lookup memberships — 200 clients per API call
    tracked = {}
    for i in range(0, len(client_ids), 200):
        batch = client_ids[i:i + 200]
        try:
            data = _api_get("client/activeclientsmemberships", [("ClientIds", cid) for cid in batch])
            for client_mem in data.get("ClientMemberships", []):
                cid = str(client_mem.get("ClientId", ""))
                memberships = client_mem.get("Memberships", [])
                for m in memberships:
                    mem_name = m.get("Name") or ""
                    if _is_tracked_membership(mem_name) and not _is_free_membership(mem_name):
                        info = client_info.get(cid, {"name": "Unknown", "status": "Unknown"})
                        tracked[cid] = {
                            "name": info["name"],
                            "membership": mem_name,
                            "status": info["status"],
                        }
                        break
        except Exception as e:
            logger.warning(f"Bulk membership lookup failed for batch: {e}")

    _cache_set("tracked_members", tracked)
    logger.info(f"Tracked members: {len(tracked)}")
    return tracked


# ── Public functions (called by tool handlers) ────────────────────────────────


def get_classes(days_back=0, days_forward=0):
    """Get class schedule for a date range.

    Args:
        days_back: Number of days back to include (0 = today only).
        days_forward: Number of days forward to include (0 = today only).
    """
    end_date = _now() + timedelta(days=days_forward)
    end = end_date.strftime("%Y-%m-%dT23:59:59")
    start_date = _now() - timedelta(days=days_back)
    start = start_date.strftime("%Y-%m-%dT00:00:00")

    all_classes = _paginated_get("class/classes", "Classes", {
        "startDateTime": start,
        "endDateTime": end,
        "hideCanceledClasses": "true",
    })

    results = []
    for cls in all_classes:
        desc = cls.get("ClassDescription", {})
        staff = cls.get("Staff", {})
        start_dt = cls.get("StartDateTime", "")
        end_dt = cls.get("EndDateTime", "")

        try:
            start_parsed = datetime.fromisoformat(start_dt.replace("Z", "+00:00"))
            start_fmt = start_parsed.strftime("%-I:%M %p")
            end_fmt = datetime.fromisoformat(end_dt.replace("Z", "+00:00")).strftime("%-I:%M %p")
            date_str = start_parsed.strftime("%a %-d %b")
        except (ValueError, TypeError):
            start_fmt = start_dt
            end_fmt = end_dt
            date_str = ""

        results.append({
            "name": desc.get("Name", "Unknown"),
            "date": date_str,
            "time": f"{start_fmt} - {end_fmt}",
            "instructor": f"{staff.get('FirstName', '')} {staff.get('LastName', '')}".strip() or "TBA",
            "booked": cls.get("TotalBooked", 0),
            "capacity": cls.get("MaxCapacity", "?"),
            "waitlisted": cls.get("TotalBookedWaitlist", 0),
        })
    return results


def get_todays_schedule():
    """Get today's class schedule (cached 1 hour)."""
    cached = _cache_get("todays_classes", ttl=CACHE_TTL_CLASSES)
    if cached is not None:
        logger.info("Using cached today's classes")
        return cached
    classes = get_classes(days_back=0)
    _cache_set("todays_classes", classes)
    return classes


def search_clients(search_text):
    """Search for clients by name, email, or phone."""
    params = {"SearchText": search_text, "limit": 20}
    data = _api_get("client/clients", params)
    clients = data.get("Clients", [])

    results = []
    for c in clients:
        results.append({
            "name": _client_name(c),
            "email": c.get("Email") or "N/A",
            "phone": c.get("MobilePhone") or c.get("HomePhone") or "N/A",
            "status": c.get("Status") or "Unknown",
            "id": c.get("Id"),
        })
    return results


# Only these membership types count as real cancellations
TRACKED_MEMBERSHIPS = [
    "all access membership - 3 month",
    "all access membership - 6 month",
    "all access membership - conversion",
    "all access membership - flex no contract",
    "conversion 12 months",
    "conversion 6 months",
    "conversion flexi",
    "post challenge",
    "student membership",
]

# Free/complimentary memberships to exclude from active count
FREE_MEMBERSHIPS = [
    "staff membership",
    "staff",
    "complimentary",
    "free",
]


def _is_free_membership(name):
    """Check if a membership is a free/staff membership."""
    if not name:
        return False
    name_lower = name.lower().strip()
    return any(f in name_lower for f in FREE_MEMBERSHIPS)


def _is_tracked_membership(name):
    """Check if a contract/membership name is one we track for cancellations."""
    if not name:
        return False
    name_lower = name.lower().strip()
    return any(m in name_lower for m in TRACKED_MEMBERSHIPS)


def _bucket_movement_by_month(signups, cancellations, window_start, window_end):
    """Bucket a list of signup/cancellation events into calendar-month buckets.

    Returns an ordered list (oldest first) of dicts:
        {"month_label": "February 2026",
         "year_month":  "2026-02",
         "partial":     bool,
         "signups":     [event, ...],
         "cancellations": [event, ...]}

    A bucket is marked `partial` when the window does not cover the full
    calendar month (e.g. window_start > 1st of month, or window_end < last of month).
    """
    from calendar import monthrange

    def _ym(date_str):
        return date_str[:7]

    def _label(ym):
        y, m = ym.split("-")
        names = ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]
        return f"{names[int(m) - 1]} {y}"

    start_ym = _ym(window_start)
    end_ym = _ym(window_end)
    months = []
    y, m = int(start_ym[:4]), int(start_ym[5:7])
    ey, em = int(end_ym[:4]), int(end_ym[5:7])
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1

    buckets = []
    for ym in months:
        year, mon = int(ym[:4]), int(ym[5:7])
        first_day = f"{ym}-01"
        last_day = f"{ym}-{monthrange(year, mon)[1]:02d}"
        partial = window_start > first_day or window_end < last_day
        buckets.append({
            "month_label": _label(ym),
            "year_month": ym,
            "partial": partial,
            "signups": [e for e in signups if _ym(e["date"]) == ym],
            "cancellations": [e for e in cancellations if _ym(e["date"]) == ym],
        })
    return buckets


def _get_client_membership_info(client_id, days_back=7):
    """Get a client's contract name and status. Only flags as tracked if the contract
    has a TerminationDate within the last N days."""
    info = {
        "contract_name": "N/A",
        "contract_status": "N/A",
        "termination_date": "N/A",
        "is_tracked": False,
    }
    if not client_id:
        return info

    cutoff = (_now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        contract_data = _api_get("client/clientcontracts", {"ClientId": client_id})
        contracts = contract_data.get("Contracts", [])
        for contract in contracts:
            contract_name = contract.get("ContractName") or "N/A"

            # Check if this contract has a recent TerminationDate
            term_date = (contract.get("TerminationDate") or "")[:10]
            if not term_date or term_date < cutoff:
                continue

            # Check if it's a tracked membership type
            if not _is_tracked_membership(contract_name):
                continue

            info["contract_name"] = contract_name
            info["contract_status"] = contract.get("AutopayStatus") or "N/A"
            info["termination_date"] = term_date
            info["is_tracked"] = True
            return info

    except Exception:
        pass

    return info


def get_member_stats():
    """Get membership statistics using cached tracked members data."""
    # Use cached tracked members (expensive lookup, cached 30 min)
    tracked = _get_tracked_members()

    # Count active tracked members by type
    active_by_type = {}
    for cid, info in tracked.items():
        if info["status"].lower() == "active":
            mem = info["membership"]
            active_by_type[mem] = active_by_type.get(mem, 0) + 1

    active_membership = sum(active_by_type.values())
    active_by_type_sorted = sorted(active_by_type.items(), key=lambda x: x[1], reverse=True)

    # Get basic status counts (cheap — just client list, no per-client lookups)
    all_clients = _get_all_clients_paginated({"IncludeInactive": "true"})

    suspended = 0
    terminated = 0
    expired = 0
    non_member = 0

    # Pre-filter: only check contracts for clients modified in last 14 days
    # (wider window than 7 days to catch edge cases)
    fourteen_days_ago = (_now() - timedelta(days=14)).strftime("%Y-%m-%dT00:00:00")
    recently_terminated = []

    for c in all_clients:
        status = (c.get("Status") or "").lower()
        if status == "suspended":
            suspended += 1
        elif status in ("terminated", "cancelled", "canceled"):
            terminated += 1
            last_modified = c.get("LastModifiedDateTime") or ""
            if last_modified and last_modified >= fourteen_days_ago[:10]:
                recently_terminated.append(c)
        elif status == "expired":
            expired += 1
        elif status in ("non-member", "non member"):
            non_member += 1

    # Now check contracts — only include those with a TerminationDate
    # on a tracked membership within the last 7 days
    recent_cancellations = []
    for c in recently_terminated:
        client_id = c.get("Id")
        membership_info = _get_client_membership_info(client_id, days_back=7)
        if membership_info.get("is_tracked"):
            recent_cancellations.append({
                "name": _client_name(c),
                "date": membership_info.get("termination_date", "N/A"),
                "contract": membership_info.get("contract_name", "N/A"),
                "contract_status": membership_info.get("contract_status", "N/A"),
            })

    # New signups in the last 30 days
    thirty_days_ago = (_now() - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00")
    new_clients = _get_all_clients_paginated(
        {"LastModifiedDate": thirty_days_ago},
        max_pages=5,
    )
    new_signups = []
    for c in new_clients:
        created = c.get("CreationDate") or ""
        if created and created >= thirty_days_ago[:10]:
            new_signups.append({
                "name": _client_name(c),
                "date": created[:10],
                "status": c.get("Status") or "Unknown",
            })

    return {
        "total_clients": len(all_clients),
        "active_membership": active_membership,
        "active_by_type": active_by_type_sorted,
        "suspended": suspended,
        "terminated": terminated,
        "expired": expired,
        "non_member": non_member,
        "new_signups_30d": len(new_signups),
        "new_signup_details": new_signups[:20],
        "recent_cancellations_7d": len(recent_cancellations),
        "recent_cancellation_details": recent_cancellations[:20],
    }


def _lookup_client_name(client_id):
    """Look up a single client's name by ID. Returns name string."""
    if not client_id:
        return "Unknown"
    try:
        data = _api_get("client/clients", {"ClientIds": client_id})
        clients = data.get("Clients", [])
        if clients:
            c = clients[0]
            return _client_name(c)
    except Exception:
        pass
    return "Unknown"


def get_payment_failures(days_back=30):
    """Get payment failures over a date range, with client names (up to 20 lookups)."""
    start = (_now() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00")
    end = _now().strftime("%Y-%m-%dT23:59:59")

    all_transactions = _paginated_get("sale/transactions", "Transactions", {
        "TransactionStartDateTime": start,
        "TransactionEndDateTime": end,
    })

    failed = [
        t for t in all_transactions
        if str(t.get("Status", "")).lower() in ("failed", "declined", "error")
    ]

    # Look up client names for failed transactions (cap at 20 to avoid API overload)
    name_cache = {}
    lookups_done = 0
    failed_details = []
    for t in failed:
        cid = t.get("ClientId", "?")
        cid_str = str(cid)
        if cid_str not in name_cache and lookups_done < 20:
            name_cache[cid_str] = _lookup_client_name(cid)
            lookups_done += 1
        failed_details.append({
            "client_id": cid,
            "client_name": name_cache.get(cid_str, "Unknown"),
            "amount": t.get("Amount", 0),
            "status": t.get("Status", "Unknown"),
            "date": (t.get("TransactionDate") or "")[:10],
        })

    return {
        "total_transactions": len(all_transactions),
        "total_failed": len(failed),
        "failed_details": failed_details,
    }


def get_revenue(days_back=7):
    """Get membership transaction revenue for the last completed Mon-Sun week."""
    today = _now()
    # Find last Monday (weekday 0 = Monday)
    days_since_monday = today.weekday()  # 0=Mon, 6=Sun
    last_monday = today - timedelta(days=days_since_monday + 7)
    last_sunday = last_monday + timedelta(days=6)

    start = last_monday.strftime("%Y-%m-%dT00:00:00")
    end = last_sunday.strftime("%Y-%m-%dT23:59:59")

    # Get all transactions in the date range
    all_transactions = _paginated_get("sale/transactions", "Transactions", {
        "TransactionStartDateTime": start,
        "TransactionEndDateTime": end,
    })

    # Filter: only count fully successful transactions
    # Exclude voided, refunded, failed, declined — including "Approved (Voided)"
    successful = []
    failed = []
    for t in all_transactions:
        status = str(t.get("Status", "")).lower()
        amount = t.get("Amount", 0) or 0
        if "void" in status or "refund" in status or status in ("failed", "declined", "error"):
            failed.append(t)
        elif amount > 0:
            successful.append(t)

    # Deduplicate by SaleId — count each sale only once
    seen_sales = set()
    unique_successful = []
    for t in successful:
        sale_id = t.get("SaleId")
        if sale_id and sale_id in seen_sales:
            continue
        if sale_id:
            seen_sales.add(sale_id)
        unique_successful.append(t)

    total_revenue = sum(t.get("Amount", 0) or 0 for t in unique_successful)

    revenue_by_day = {}
    for t in unique_successful:
        txn_date = (t.get("TransactionTime") or "")[:10]
        if txn_date:
            amount = t.get("Amount", 0) or 0
            revenue_by_day[txn_date] = revenue_by_day.get(txn_date, 0) + amount

    sorted_days = sorted(revenue_by_day.items())

    return {
        "total_revenue": total_revenue,
        "members_debited": len({t.get("ClientId") for t in unique_successful}),
        "revenue_by_day": sorted_days,
        "start_date": start[:10],
        "end_date": end[:10],
    }


def format_revenue(rev):
    """Format membership revenue report for WhatsApp."""
    lines = [
        f"*Membership Revenue Report*",
        f"*Week: {rev['start_date']} to {rev['end_date']}*\n",
        f"Total revenue: ${rev['total_revenue']:,.2f}",
        f"Members debited: {rev.get('members_debited', 0)}",
    ]

    if rev["revenue_by_day"]:
        lines.append(f"\n*Daily Breakdown:*")
        for day, amount in rev["revenue_by_day"]:
            lines.append(f"  {day}: ${amount:,.2f}")

    return "\n".join(lines)


def get_daily_briefing():
    """Get today's class schedule for the daily briefing."""
    classes = get_todays_schedule()
    total_bookings = sum(c["booked"] for c in classes)

    return {
        "date": _now().strftime("%A, %B %-d, %Y"),
        "classes": classes,
        "total_classes": len(classes),
        "total_bookings": total_bookings,
    }


# ── Formatters ────────────────────────────────────────────────────────────────


def format_schedule(classes):
    """Format class schedule for WhatsApp."""
    if not classes:
        return "No classes scheduled today."

    today_str = _now().strftime("%A %-d %B")
    lines = [f"*Classes - {today_str}*\n"]
    for c in classes:
        wl = f" (+{c['waitlisted']} waitlist)" if c['waitlisted'] else ""
        date_line = f"  Date: {c['date']}\n" if c.get('date') else ""
        lines.append(
            f"*{c['name']}*\n"
            f"{date_line}"
            f"  {c['time']}\n"
            f"  Instructor: {c['instructor']}\n"
            f"  Booked: {c['booked']}/{c['capacity']}{wl}"
        )

    total = sum(c["booked"] for c in classes)
    lines.append(f"\n{len(classes)} classes, {total} total bookings")
    return "\n\n".join(lines)


def format_briefing(briefing):
    """Format daily briefing (classes only) for WhatsApp."""
    lines = [f"*HIIT Station Capalaba*\n*Daily Briefing - {briefing['date']}*\n"]

    lines.append(f"*TODAY'S CLASSES ({briefing['total_classes']})*")
    for c in briefing["classes"]:
        lines.append(f"  {c['name']}\n  {c['time']} - {c['instructor']} - {c['booked']}/{c['capacity']}")
    lines.append(f"\nTotal bookings: {briefing['total_bookings']}")

    return "\n".join(lines)


def format_member_stats(stats):
    """Format member stats for WhatsApp."""
    lines = [
        "*Member Statistics*\n",
        f"  Active memberships: {stats['active_membership']}\n",
        f"  *Breakdown by type:*",
    ]
    if stats.get("active_by_type"):
        for name, count in stats["active_by_type"]:
            lines.append(f"  - {name}: {count}")
    else:
        lines.append("  No breakdown available")
    lines += [
        f"\n  Suspended: {stats['suspended']}",
        f"  Expired: {stats['expired']}",
        f"\n*Membership Cancellations (Last 7 Days): {stats['recent_cancellations_7d']}*",
    ]
    if stats["recent_cancellations_7d"] == 0:
        lines.append("  None")
    for c in stats["recent_cancellation_details"]:
        lines.append(f"  {c['name']} - {c['contract']} - {c['date']}")
    if stats["recent_cancellations_7d"] > 20:
        lines.append(f"  ... and {stats['recent_cancellations_7d'] - 20} more")

    lines.append(f"\n*New Signups (Last 30 Days): {stats['new_signups_30d']}*")
    for s in stats["new_signup_details"]:
        lines.append(f"  {s['name']} - {s['date']} [{s['status']}]")
    if stats["new_signups_30d"] > 20:
        lines.append(f"  ... and {stats['new_signups_30d'] - 20} more")

    return "\n".join(lines)


def format_payment_failures(payments):
    """Format payment failures for WhatsApp."""
    p = payments
    lines = [
        f"*Payment Report (Last 30 Days)*\n",
        f"Total transactions: {p['total_transactions']}",
    ]
    if p["total_failed"] == 0:
        lines.append("Payment failures: None")
    else:
        lines.append(f"Payment failures: {p['total_failed']}\n")
        for f in p["failed_details"][:15]:
            name = f.get("client_name", "Unknown")
            lines.append(f"  {name} (ID {f['client_id']}) - ${f['amount']:.2f} - {f['status']} ({f['date']})")
        if p["total_failed"] > 15:
            lines.append(f"  ... and {p['total_failed'] - 15} more")

    return "\n".join(lines)


def format_clients(clients):
    """Format client search results for WhatsApp."""
    if not clients:
        return "No clients found."

    lines = [f"*Client Search Results ({len(clients)})*\n"]
    for i, c in enumerate(clients, 1):
        lines.append(f"{i}. *{c['name']}*\n   {c['email']}\n   {c['phone']}\n   {c['status']}")
    return "\n".join(lines)


# ── New Members ────────────────────────────────────────────────────────────────


def get_new_members(days_back=7):
    """Get new member sign-ups from MindBody for the last N days.

    Fetches recently modified clients, filters by CreationDate to find
    genuinely new clients, then looks up their membership type.
    """
    cache_key = f"new_members_{days_back}"
    cached = _cache_get(cache_key, ttl=CACHE_TTL_CLASSES)
    if cached is not None:
        logger.info(f"Using cached new members ({days_back}d)")
        return cached

    cutoff = (_now() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00")
    cutoff_date = cutoff[:10]

    # Fetch recently modified clients
    clients = _get_all_clients_paginated(
        {"LastModifiedDate": cutoff},
        max_pages=5,
    )

    # Filter to genuinely new clients (created within the window)
    new_clients = []
    for c in clients:
        created = c.get("CreationDate") or ""
        if created and created[:10] >= cutoff_date:
            new_clients.append(c)

    # Look up memberships in bulk (batches of 200)
    client_ids = [str(c.get("Id")) for c in new_clients if c.get("Id")]
    membership_map = {}  # client_id -> membership name
    for i in range(0, len(client_ids), 200):
        batch = client_ids[i:i + 200]
        try:
            data = _api_get("client/activeclientmemberships", [("ClientIds", cid) for cid in batch])
            for cm in data.get("ClientMemberships", []):
                cid = str(cm.get("ClientId", ""))
                memberships = cm.get("Memberships", [])
                if memberships:
                    membership_map[cid] = memberships[0].get("Name") or "N/A"
        except Exception as e:
            logger.warning(f"Membership lookup failed for new members batch: {e}")

    results = []
    for c in new_clients:
        cid = str(c.get("Id", ""))
        results.append({
            "name": _client_name(c),
            "date_signed_up": (c.get("CreationDate") or "")[:10],
            "membership_type": membership_map.get(cid, "N/A"),
            "email": c.get("Email") or "N/A",
            "phone": c.get("MobilePhone") or c.get("HomePhone") or "N/A",
        })

    # Sort by date descending
    results.sort(key=lambda x: x["date_signed_up"], reverse=True)
    _cache_set(cache_key, results)
    return results


def format_new_members(members, days_back=7):
    """Format new members list for WhatsApp."""
    if not members:
        return f"*New Members (Last {days_back} Days)*\n\nNo new sign-ups found."

    lines = [f"*New Members (Last {days_back} Days)*\n"]
    lines.append(f"Total: {len(members)} new sign-ups\n")
    for i, m in enumerate(members, 1):
        lines.append(
            f"{i}. *{m['name']}*\n"
            f"   Signed up: {m['date_signed_up']}\n"
            f"   Membership: {m['membership_type']}\n"
            f"   Email: {m['email']}\n"
            f"   Phone: {m['phone']}"
        )
    return "\n\n".join(lines)


# ── Membership Movement Report ────────────────────────────────────────────────

WHATSAPP_MAX_CHARS = 1500


def format_membership_movement(result, days_back=90, split_by_month=False):
    """Format a membership movement result dict for WhatsApp.

    Flat mode: grouped by membership with names underneath.
    Monthly mode: same groupings bucketed into calendar months (see Task 4).
    """
    signups = result.get("signups", [])
    cancellations = result.get("cancellations", [])
    net = len(signups) - len(cancellations)
    net_str = f"+{net}" if net > 0 else str(net)

    if split_by_month:
        return _format_membership_movement_monthly(result, days_back, net_str)

    lines = [f"*Membership Report — Last {days_back} Days*", ""]
    lines.append(f"*SIGNUPS: {len(signups)}*")
    lines.extend(_format_group_block(signups))
    lines.append("")
    lines.append(f"*CANCELLATIONS: {len(cancellations)}*")
    lines.extend(_format_group_block(cancellations))
    lines.append("")
    lines.append(f"*Net: {net_str}*")

    return _truncate_to_whatsapp("\n".join(lines))


def _format_group_block(events):
    """Group events by membership name, sort by count desc, render bullets."""
    if not events:
        return ["(none)"]
    by_mem = {}
    for e in events:
        by_mem.setdefault(e["membership"], []).append(e)
    ordered = sorted(by_mem.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    out = []
    for mem_name, group in ordered:
        out.append(f"• {mem_name} — {len(group)}")
        for e in sorted(group, key=lambda x: x["date"], reverse=True):
            out.append(f"   - {e['name']} ({e['date']})")
    return out


def _truncate_to_whatsapp(text):
    """If text exceeds WHATSAPP_MAX_CHARS, truncate and append a notice."""
    if len(text) <= WHATSAPP_MAX_CHARS:
        return text
    cutoff = WHATSAPP_MAX_CHARS - 40
    return text[:cutoff].rstrip() + "\n… (truncated — ask for a month)"


def _format_membership_movement_monthly(result, days_back, net_str):
    signups = result.get("signups", [])
    cancellations = result.get("cancellations", [])
    buckets = _bucket_movement_by_month(
        signups, cancellations,
        window_start=result["window_start"],
        window_end=result["window_end"],
    )

    lines = [f"*Membership Report — Last {days_back} Days (by month)*", ""]
    for b in buckets:
        label = b["month_label"] + (" (partial)" if b["partial"] else "")
        bucket_net = len(b["signups"]) - len(b["cancellations"])
        bucket_net_str = f"+{bucket_net}" if bucket_net > 0 else str(bucket_net)
        lines.append(f"*── {label} ──*")
        lines.append(
            f"Signups: {len(b['signups'])}   "
            f"Cancellations: {len(b['cancellations'])}   "
            f"Net: {bucket_net_str}"
        )
        if b["signups"]:
            lines.append("  Signups:")
            for sub in _summarise_by_membership(b["signups"]):
                lines.append(f"   • {sub}")
        if b["cancellations"]:
            lines.append("  Cancellations:")
            for sub in _summarise_by_membership(b["cancellations"]):
                lines.append(f"   • {sub}")
        lines.append("")

    lines.append(
        f"*Totals — Signups: {len(signups)} · "
        f"Cancellations: {len(cancellations)} · Net: {net_str}*"
    )
    return _truncate_to_whatsapp("\n".join(lines))


def _summarise_by_membership(events):
    """Render a compact one-line-per-membership summary: 'Name: A, B, C'."""
    by_mem = {}
    for e in events:
        by_mem.setdefault(e["membership"], []).append(e)
    ordered = sorted(by_mem.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    lines = []
    for mem_name, group in ordered:
        names = ", ".join(
            e["name"] for e in sorted(group, key=lambda x: x["date"], reverse=True)
        )
        lines.append(f"{mem_name}: {names}")
    return lines


def get_membership_movement(days_back=90):
    """Return signups and cancellations of tracked memberships over the last N days.

    Signup  = tracked contract with StartDate within the window.
    Cancel  = tracked contract with TerminationDate within the window.
    Only contracts matching TRACKED_MEMBERSHIPS count. A single contract can
    appear in both lists if it both started and terminated in the window.

    Cached for 1 hour per days_back value.
    """
    cache_key = f"membership_movement_{days_back}"
    cached = _cache_get(cache_key, ttl=CACHE_TTL_CLASSES)
    if cached is not None:
        logger.info(f"Using cached membership movement ({days_back}d)")
        return cached

    window_end = _now().strftime("%Y-%m-%d")
    window_start = (_now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    modified_since = (_now() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00")

    candidates = _get_all_clients_paginated(
        {"LastModifiedDate": modified_since, "IncludeInactive": "true"},
        max_pages=15,
    )

    signups = []
    cancellations = []
    seen_signup = set()
    seen_cancellation = set()

    for c in candidates:
        client_id = c.get("Id")
        if not client_id:
            continue
        try:
            contract_data = _api_get("client/clientcontracts", {"ClientId": client_id})
        except Exception as e:
            logger.warning(f"clientcontracts lookup failed for {client_id}: {e}")
            continue

        for contract in contract_data.get("Contracts", []) or []:
            contract_name = contract.get("ContractName") or ""
            if not _is_tracked_membership(contract_name):
                continue

            contract_id = contract.get("Id") or contract.get("ContractId") or 0
            client_name = _client_name(c)

            start_date = (contract.get("StartDate") or "")[:10]
            if start_date and window_start <= start_date <= window_end:
                key = (client_id, contract_id)
                if key not in seen_signup:
                    seen_signup.add(key)
                    signups.append({
                        "client_id": client_id,
                        "contract_id": contract_id,
                        "name": client_name,
                        "membership": contract_name,
                        "date": start_date,
                    })

            term_date = (contract.get("TerminationDate") or "")[:10]
            if term_date and window_start <= term_date <= window_end:
                key = (client_id, contract_id)
                if key not in seen_cancellation:
                    seen_cancellation.add(key)
                    cancellations.append({
                        "client_id": client_id,
                        "contract_id": contract_id,
                        "name": client_name,
                        "membership": contract_name,
                        "date": term_date,
                    })

    signups.sort(key=lambda x: x["date"], reverse=True)
    cancellations.sort(key=lambda x: x["date"], reverse=True)

    result = {
        "days_back": days_back,
        "window_start": window_start,
        "window_end": window_end,
        "signups": signups,
        "cancellations": cancellations,
    }
    _cache_set(cache_key, result)
    logger.info(
        f"Membership movement {days_back}d: "
        f"{len(signups)} signups, {len(cancellations)} cancellations "
        f"(scanned {len(candidates)} clients)"
    )
    return result


# ── Arrears Report ─────────────────────────────────────────────────────────────


def get_arrears_report():
    """Pull failed/declined transactions for the last 30 days, grouped by client.

    This is the Thursday arrears report — shows total owed per client.
    """
    cached = _cache_get("arrears_report", ttl=CACHE_TTL_CLASSES)
    if cached is not None:
        logger.info("Using cached arrears report")
        return cached

    start = (_now() - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00")
    end = _now().strftime("%Y-%m-%dT23:59:59")

    all_transactions = _paginated_get("sale/transactions", "Transactions", {
        "TransactionStartDateTime": start,
        "TransactionEndDateTime": end,
    })

    # Filter failed/declined transactions
    failed = [
        t for t in all_transactions
        if str(t.get("Status", "")).lower() in ("failed", "declined", "error")
    ]

    # Group by client
    by_client = {}
    for t in failed:
        cid = str(t.get("ClientId", "?"))
        amount = t.get("Amount", 0) or 0
        if cid not in by_client:
            by_client[cid] = {"total_owed": 0, "transactions": []}
        by_client[cid]["total_owed"] += amount
        by_client[cid]["transactions"].append({
            "amount": amount,
            "date": (t.get("TransactionDate") or "")[:10],
            "status": t.get("Status", "Unknown"),
        })

    # Look up client names (cap at 20 lookups)
    lookups_done = 0
    arrears = []
    for cid, info in sorted(by_client.items(), key=lambda x: x[1]["total_owed"], reverse=True):
        name = "Unknown"
        if lookups_done < 20:
            name = _lookup_client_name(cid)
            lookups_done += 1
        arrears.append({
            "client_id": cid,
            "client_name": name,
            "total_owed": info["total_owed"],
            "failed_count": len(info["transactions"]),
        })

    result = {
        "total_clients_in_arrears": len(arrears),
        "total_amount_owed": sum(a["total_owed"] for a in arrears),
        "arrears_details": arrears,
        "period": f"{start[:10]} to {end[:10]}",
    }
    _cache_set("arrears_report", result)
    return result


def format_arrears_report(report):
    """Format arrears report for WhatsApp (Thursday report)."""
    lines = [
        f"*ARREARS REPORT*",
        f"*Period: {report['period']}*\n",
        f"Clients in arrears: {report['total_clients_in_arrears']}",
        f"Total outstanding: ${report['total_amount_owed']:,.2f}\n",
    ]

    if not report["arrears_details"]:
        lines.append("No failed payments found.")
    else:
        for i, a in enumerate(report["arrears_details"][:20], 1):
            lines.append(
                f"{i}. *{a['client_name']}* (ID {a['client_id']})\n"
                f"   Owed: ${a['total_owed']:,.2f} ({a['failed_count']} failed payment{'s' if a['failed_count'] != 1 else ''})"
            )
        if report["total_clients_in_arrears"] > 20:
            lines.append(f"\n... and {report['total_clients_in_arrears'] - 20} more")

    return "\n".join(lines)


# ── Weekly Summary ─────────────────────────────────────────────────────────────


def get_weekly_summary():
    """Generate a Friday weekly wrap-up: classes, bookings, revenue, signups, cancellations.

    Covers last completed Mon-Sun week.
    """
    cached = _cache_get("weekly_summary", ttl=CACHE_TTL_CLASSES)
    if cached is not None:
        logger.info("Using cached weekly summary")
        return cached

    today = _now()
    # Find last Monday (weekday 0 = Monday)
    days_since_monday = today.weekday()
    last_monday = today - timedelta(days=days_since_monday + 7)
    last_sunday = last_monday + timedelta(days=6)

    start = last_monday.strftime("%Y-%m-%dT00:00:00")
    end = last_sunday.strftime("%Y-%m-%dT23:59:59")

    # 1. Classes and bookings
    all_classes = _paginated_get("class/classes", "Classes", {
        "startDateTime": start,
        "endDateTime": end,
        "hideCanceledClasses": "true",
    })

    total_classes = len(all_classes)
    total_bookings = sum(c.get("TotalBooked", 0) for c in all_classes)

    # 2. Revenue (reuse get_revenue which does last Mon-Sun)
    rev = get_revenue()

    # 3. New signups in the week
    cutoff = last_monday.strftime("%Y-%m-%dT00:00:00")
    cutoff_date = cutoff[:10]
    end_date = last_sunday.strftime("%Y-%m-%d")

    clients = _get_all_clients_paginated(
        {"LastModifiedDate": cutoff},
        max_pages=5,
    )
    new_signups = 0
    for c in clients:
        created = (c.get("CreationDate") or "")[:10]
        if created and cutoff_date <= created <= end_date:
            new_signups += 1

    # 4. Cancellations in the week (terminated/cancelled with LastModified in range)
    cancellations = 0
    for c in clients:
        status = (c.get("Status") or "").lower()
        last_mod = (c.get("LastModifiedDateTime") or "")[:10]
        if status in ("terminated", "cancelled", "canceled") and last_mod and cutoff_date <= last_mod <= end_date:
            cancellations += 1

    result = {
        "week_start": last_monday.strftime("%a %-d %b"),
        "week_end": last_sunday.strftime("%a %-d %b"),
        "total_classes": total_classes,
        "total_bookings": total_bookings,
        "total_revenue": rev.get("total_revenue", 0),
        "members_debited": rev.get("members_debited", 0),
        "new_signups": new_signups,
        "cancellations": cancellations,
    }
    _cache_set("weekly_summary", result)
    return result


def format_weekly_summary(summary):
    """Format weekly summary for WhatsApp — ready to copy-paste to staff chat."""
    lines = [
        f"*WEEKLY WRAP-UP*",
        f"*{summary['week_start']} - {summary['week_end']}*\n",
        f"*Classes & Bookings*",
        f"  Total classes: {summary['total_classes']}",
        f"  Total bookings: {summary['total_bookings']}",
        f"  Avg per class: {summary['total_bookings'] / max(summary['total_classes'], 1):.1f}\n",
        f"*Revenue*",
        f"  Total: ${summary['total_revenue']:,.2f}",
        f"  Members debited: {summary['members_debited']}\n",
        f"*Membership*",
        f"  New sign-ups: {summary['new_signups']}",
        f"  Cancellations: {summary['cancellations']}",
    ]
    return "\n".join(lines)


# ── Class Report (New/Intro Client Check) ─────────────────────────────────────

INTRO_KEYWORDS = [
    "intro", "trial", "offer", "taster", "first timer", "starter",
    "21 day", "21-day", "7 day", "7-day", "14 day", "14-day",
    "free", "kickstart", "2 week", "2-week",
    "bring a friend", "casual pass", "casual",
]


def _is_intro_pricing(name):
    """Check if a pricing option name is an intro/trial offer."""
    if not name:
        return False
    name_lower = name.lower()
    return any(kw in name_lower for kw in INTRO_KEYWORDS)


def _normalize_time(time_str):
    """Normalize a time string like '6am', '6:00 AM', '18:00', '6pm' to HH:MM 24hr format."""
    if not time_str:
        return None
    t = time_str.strip().lower().replace(" ", "")

    # Already HH:MM 24hr
    if len(t) in (4, 5) and t.replace(":", "").isdigit():
        if ":" in t:
            return t.zfill(5)  # "6:00" -> "06:00"
        else:
            return t[:2] + ":" + t[2:]  # "0600" -> "06:00"

    # Parse am/pm formats: "6am", "6pm", "6:00am", "630pm"
    is_pm = "pm" in t
    is_am = "am" in t
    t = t.replace("am", "").replace("pm", "").strip()

    if not (is_am or is_pm):
        return time_str.strip()  # Can't parse, return as-is

    if ":" in t:
        parts = t.split(":")
        hour, minute = int(parts[0]), int(parts[1])
    elif len(t) <= 2:
        hour, minute = int(t), 0
    elif len(t) in (3, 4):
        hour, minute = int(t[:-2]), int(t[-2:])
    else:
        return time_str.strip()

    if is_pm and hour != 12:
        hour += 12
    elif is_am and hour == 12:
        hour = 0

    return f"{hour:02d}:{minute:02d}"


def _get_day_classes(class_date=None):
    """Fetch all classes for a given date from MindBody. Returns list of API class objects."""
    if not class_date:
        class_date = _now().strftime("%Y-%m-%d")
    start = f"{class_date}T00:00:00"
    end = f"{class_date}T23:59:59"
    data = _api_get("class/classes", {
        "startDateTime": start,
        "endDateTime": end,
        "hideCanceledClasses": "true",
        "limit": 200,
    })
    return data.get("Classes", [])


def _cls_to_dict(cls, class_date):
    """Convert a raw MindBody class object to our standard class_info dict."""
    desc = cls.get("ClassDescription", {})
    staff = cls.get("Staff", {})
    try:
        time_fmt = datetime.fromisoformat(
            cls.get("StartDateTime", "").replace("Z", "+00:00")
        ).strftime("%-I:%M %p")
    except (ValueError, TypeError):
        time_fmt = "?"
    return {
        "id": cls.get("Id"),
        "name": desc.get("Name", "Unknown"),
        "time_fmt": time_fmt,
        "date": class_date,
        "instructor": f"{staff.get('FirstName', '')} {staff.get('LastName', '')}".strip() or "TBA",
        "total_booked": cls.get("TotalBooked", 0),
        "raw": cls,
    }


def _find_class(class_name, class_date=None, class_time=None):
    """Find a class by name, date, and optional time. Returns (class_dict, error_dict).

    On success: (class_info, None)  where class_info has keys:
        id, name, time_fmt, date, instructor, total_booked, raw (original API object)
    On error: (None, {"error": "..."})
    """
    if not class_date:
        class_date = _now().strftime("%Y-%m-%d")

    # Normalize time to HH:MM 24hr
    class_time = _normalize_time(class_time)

    classes = _get_day_classes(class_date)

    matched = []
    for cls in classes:
        desc = cls.get("ClassDescription", {})
        name = desc.get("Name", "")
        if class_name.lower() in name.lower():
            if class_time:
                cls_time = (cls.get("StartDateTime") or "")[11:16]
                if cls_time == class_time or cls_time.replace(":", "") == class_time.replace(":", ""):
                    matched.append(cls)
            else:
                matched.append(cls)

    if not matched:
        return None, {"error": f"No class matching '{class_name}' found on {class_date}"}

    if len(matched) > 1 and not class_time:
        options = []
        for cls in matched:
            desc = cls.get("ClassDescription", {})
            try:
                t = datetime.fromisoformat(cls.get("StartDateTime", "").replace("Z", "+00:00")).strftime("%-I:%M %p")
            except (ValueError, TypeError):
                t = cls.get("StartDateTime", "?")
            options.append(f"{desc.get('Name', '?')} at {t}")
        return None, {"error": f"Multiple classes found. Which one?\n" + "\n".join(f"  - {o}" for o in options)}

    target = matched[0]
    return _cls_to_dict(target, class_date), None


def _find_all_classes(class_date=None, class_time=None):
    """Find all classes for a date, optionally filtered by time.

    Returns list of class_info dicts (same shape as _find_class success result).
    """
    if not class_date:
        class_date = _now().strftime("%Y-%m-%d")

    class_time = _normalize_time(class_time)
    classes = _get_day_classes(class_date)

    results = []
    for cls in classes:
        if class_time:
            cls_time = (cls.get("StartDateTime") or "")[11:16]
            if cls_time != class_time and cls_time.replace(":", "") != class_time.replace(":", ""):
                continue
        results.append(_cls_to_dict(cls, class_date))

    return results


def _get_class_visits(class_id):
    """Fetch the visit roster for a class. Returns list of visit dicts."""
    try:
        visits_data = _api_get("class/classvisits", {"classID": class_id})
        if visits_data and isinstance(visits_data, dict):
            return (visits_data.get("Class") or {}).get("Visits") or []
    except Exception:
        pass
    return []


def run_class_report(class_name, class_date=None, class_time=None):
    """Run a new client report for a specific class.

    Checks each booked client for:
    1. First class ever (visit count <= 1)
    2. On an intro/trial pricing option (from roster ServiceName)
    3. New membership contract (signed in last 14 days)
    """
    cls, error = _find_class(class_name, class_date, class_time)
    if error:
        return error

    class_name_full = cls["name"]
    class_time_fmt = cls["time_fmt"]
    class_date = cls["date"]
    instructor = cls["instructor"]
    total_booked = cls["total_booked"]
    class_id = cls["id"]

    # Step 2: Pull the roster
    visits = _get_class_visits(class_id)

    if not visits:
        return {
            "class_name": class_name_full,
            "class_time": class_time_fmt,
            "class_date": class_date,
            "instructor": instructor,
            "total_booked": total_booked,
            "flagged_clients": [],
            "total_checked": 0,
        }

    # Step 3: Bulk-fetch all client info in one API call (instead of per-client)
    fourteen_days_ago = (_now() - timedelta(days=14)).strftime("%Y-%m-%d")
    two_years_ago = (_now() - timedelta(days=730)).strftime("%Y-%m-%dT00:00:00")
    end_now = _now().strftime("%Y-%m-%dT23:59:59")
    flagged = []

    # Collect all client IDs from the roster
    client_ids = [str(v.get("ClientId", "")) for v in visits if v.get("ClientId")]

    # Pass 1: Use FREE roster data to find candidates (no API calls).
    # Only candidates get the per-client lookups — this is the key optimisation.
    candidates = []
    for v in visits:
        cid = str(v.get("ClientId", ""))
        if not cid:
            continue

        service_obj = v.get("Service") or {}
        service_name = v.get("ServiceName") or service_obj.get("Name") or ""
        is_intro = _is_intro_pricing(service_name)

        # If they're on intro/trial pricing → candidate (no API call needed to decide)
        if is_intro:
            candidates.append({
                "cid": cid,
                "service_name": service_name,
                "is_intro": True,
            })

    logger.info(f"Class report: {len(visits)} visits, {len(candidates)} intro candidates")

    # Pass 2: For each candidate, fetch name + visit count + contract (3 calls each)
    for c in candidates:
        cid = c["cid"]

        # Client name + creation date (1 API call)
        client_name = "Unknown"
        is_new_account = False
        try:
            client_data = _api_get("client/clients", {"ClientIds": cid})
            clients_list = (client_data or {}).get("Clients") or []
            if clients_list:
                client_name = _client_name(clients_list[0])
                created = (clients_list[0].get("CreationDate") or "")[:10]
                is_new_account = created >= fourteen_days_ago if created else False
        except Exception as e:
            logger.warning(f"Client lookup failed for {cid}: {e}")

        # Visit history (1 API call)
        visit_count = 0
        try:
            visit_data = _api_get("client/clientvisits", {
                "ClientId": cid,
                "StartDate": two_years_ago,
                "EndDate": end_now,
            })
            past_visits = (visit_data or {}).get("Visits") or []
            visit_count = len(past_visits)
        except Exception:
            pass

        # Contract lookup (1 API call)
        new_membership = False
        membership_name = None
        membership_date = None
        try:
            contract_data = _api_get("client/clientcontracts", {"ClientId": cid})
            contracts = (contract_data or {}).get("Contracts") or []
            for con in contracts:
                agreement_date = (con.get("AgreementDate") or "")[:10]
                if agreement_date and agreement_date >= fourteen_days_ago:
                    new_membership = True
                    membership_name = con.get("ContractName") or "Unknown"
                    membership_date = agreement_date
                    break
        except Exception:
            pass

        # Determine class status label
        is_new_client = visit_count <= 2 or is_new_account
        if visit_count == 0:
            class_status = "1st class ever"
        elif visit_count == 1:
            class_status = "2nd class"
        elif visit_count == 2:
            class_status = "3rd class"
        elif is_new_account:
            class_status = f"New client ({visit_count + 1} total visits)"
        else:
            class_status = None

        if is_new_client or c["is_intro"] or new_membership:
            flagged.append({
                "name": client_name,
                "pricing_option": c["service_name"],
                "class_status": class_status,
                "intro_offer": c["is_intro"],
                "intro_offer_name": c["service_name"] if c["is_intro"] else None,
                "new_membership": new_membership,
                "membership_name": membership_name,
                "membership_date": membership_date,
            })

    return {
        "class_name": class_name_full,
        "class_time": class_time_fmt,
        "class_date": class_date,
        "instructor": instructor,
        "total_booked": total_booked,
        "flagged_clients": flagged,
        "total_checked": len(visits),
    }


def format_class_report(report):
    """Format class report for WhatsApp."""
    if "error" in report:
        return report["error"]

    lines = [
        f"*New Client Report*",
        f"*{report['class_name']}* — {report['class_date']} at {report['class_time']}",
        f"Instructor: {report['instructor']}",
        f"Total booked: {report['total_booked']}\n",
    ]

    flagged = report.get("flagged_clients", [])
    if not flagged:
        lines.append("No new or intro clients found in this class.")
    else:
        for i, c in enumerate(flagged, 1):
            lines.append(f"*{i}. {c['name']}*")
            lines.append(f"  Pricing: {c.get('pricing_option', 'N/A')}")
            if c.get("class_status"):
                lines.append(f"  {c['class_status'].upper()}")
            if c["intro_offer"]:
                lines.append(f"  Intro/trial: Yes")
            if c["new_membership"]:
                lines.append(f"  New membership: {c['membership_name']} (started {c['membership_date']})")
            lines.append("")

        lines.append(f"{len(flagged)} new/intro clients out of {report['total_checked']} booked.")

    return "\n".join(lines)


def run_multi_class_report(class_date=None, class_time=None):
    """Run class reports for all classes on a date (optionally filtered by time)."""
    all_classes = _find_all_classes(class_date=class_date, class_time=class_time)
    if not all_classes:
        date_str = class_date or _now().strftime("%Y-%m-%d")
        return [{"error": f"No classes found on {date_str}"}]

    reports = []
    for cls in all_classes:
        report = _run_class_report_for(cls)
        reports.append(report)
    return reports


def _run_class_report_for(cls):
    """Run class report for a pre-resolved class dict (from _find_all_classes)."""
    class_id = cls["id"]
    class_date = cls["date"]

    visits = _get_class_visits(class_id)

    if not visits:
        return {
            "class_name": cls["name"],
            "class_time": cls["time_fmt"],
            "class_date": class_date,
            "instructor": cls["instructor"],
            "total_booked": cls["total_booked"],
            "flagged_clients": [],
            "total_checked": 0,
        }

    fourteen_days_ago = (_now() - timedelta(days=14)).strftime("%Y-%m-%d")
    two_years_ago = (_now() - timedelta(days=730)).strftime("%Y-%m-%dT00:00:00")
    end_now = _now().strftime("%Y-%m-%dT23:59:59")
    flagged = []

    candidates = []
    for v in visits:
        cid = str(v.get("ClientId", ""))
        if not cid:
            continue
        service_obj = v.get("Service") or {}
        service_name = v.get("ServiceName") or service_obj.get("Name") or ""
        is_intro = _is_intro_pricing(service_name)
        if is_intro:
            candidates.append({"cid": cid, "service_name": service_name, "is_intro": True})

    for c in candidates:
        cid = c["cid"]
        client_name = "Unknown"
        is_new_account = False
        try:
            client_data = _api_get("client/clients", {"ClientIds": cid})
            clients_list = (client_data or {}).get("Clients") or []
            if clients_list:
                client_name = _client_name(clients_list[0])
                created = (clients_list[0].get("CreationDate") or "")[:10]
                is_new_account = created >= fourteen_days_ago if created else False
        except Exception as e:
            logger.warning(f"Client lookup failed for {cid}: {e}")

        visit_count = 0
        try:
            visit_data = _api_get("client/clientvisits", {
                "ClientId": cid, "StartDate": two_years_ago, "EndDate": end_now,
            })
            past_visits = (visit_data or {}).get("Visits") or []
            visit_count = len(past_visits)
        except Exception:
            pass

        new_membership = False
        membership_name = None
        membership_date = None
        try:
            contract_data = _api_get("client/clientcontracts", {"ClientId": cid})
            contracts = (contract_data or {}).get("Contracts") or []
            for con in contracts:
                agreement_date = (con.get("AgreementDate") or "")[:10]
                if agreement_date and agreement_date >= fourteen_days_ago:
                    new_membership = True
                    membership_name = con.get("ContractName") or "Unknown"
                    membership_date = agreement_date
                    break
        except Exception:
            pass

        is_new_client = visit_count <= 2 or is_new_account
        if visit_count == 0:
            class_status = "1st class ever"
        elif visit_count == 1:
            class_status = "2nd class"
        elif visit_count == 2:
            class_status = "3rd class"
        elif is_new_account:
            class_status = f"New client ({visit_count + 1} total visits)"
        else:
            class_status = None

        if is_new_client or c["is_intro"] or new_membership:
            flagged.append({
                "name": client_name,
                "pricing_option": c["service_name"],
                "class_status": class_status,
                "intro_offer": c["is_intro"],
                "intro_offer_name": c["service_name"] if c["is_intro"] else None,
                "new_membership": new_membership,
                "membership_name": membership_name,
                "membership_date": membership_date,
            })

    return {
        "class_name": cls["name"],
        "class_time": cls["time_fmt"],
        "class_date": class_date,
        "instructor": cls["instructor"],
        "total_booked": cls["total_booked"],
        "flagged_clients": flagged,
        "total_checked": len(visits),
    }


def format_multi_class_report(reports):
    """Format multiple class reports into one WhatsApp message."""
    if len(reports) == 1 and "error" in reports[0]:
        return reports[0]["error"]

    parts = []
    for report in reports:
        parts.append(format_class_report(report))

    return "\n\n---\n\n".join(parts)


# ── No-Show Report ────────────────────────────────────────────────────────────


def get_noshow_report(class_name, class_date=None, class_time=None):
    """Get clients who were booked into a class but did not sign in.

    Uses the SignedIn field from classvisits — only meaningful after a class
    has started/finished. Also excludes late cancellations.
    """
    cls, error = _find_class(class_name, class_date, class_time)
    if error:
        return error

    visits = _get_class_visits(cls["id"])

    if not visits:
        return {
            "class_name": cls["name"],
            "class_time": cls["time_fmt"],
            "class_date": cls["date"],
            "instructor": cls["instructor"],
            "total_booked": cls["total_booked"],
            "no_shows": [],
            "signed_in_count": 0,
        }

    no_shows = []
    signed_in_count = 0

    for v in visits:
        # Skip late cancellations — they freed the spot
        if v.get("LateCancelled") or v.get("LateCanceled"):
            continue

        signed_in = v.get("SignedIn", False)
        if signed_in:
            signed_in_count += 1
            continue

        cid = str(v.get("ClientId", ""))
        if not cid:
            continue

        client_name = _lookup_client_name(cid)
        service_obj = v.get("Service") or {}
        service_name = v.get("ServiceName") or service_obj.get("Name") or "N/A"

        no_shows.append({
            "name": client_name,
            "client_id": cid,
            "pricing": service_name,
        })

    return {
        "class_name": cls["name"],
        "class_time": cls["time_fmt"],
        "class_date": cls["date"],
        "instructor": cls["instructor"],
        "total_booked": cls["total_booked"],
        "no_shows": no_shows,
        "signed_in_count": signed_in_count,
    }


def format_noshow_report(report):
    """Format no-show report for WhatsApp."""
    if "error" in report:
        return report["error"]

    lines = [
        f"*No-Show Report*",
        f"*{report['class_name']}* — {report['class_date']} at {report['class_time']}",
        f"Instructor: {report['instructor']}",
        f"Booked: {report['total_booked']}, Signed in: {report['signed_in_count']}\n",
    ]

    no_shows = report.get("no_shows", [])
    if not no_shows:
        lines.append("Everyone signed in!")
    else:
        lines.append(f"*NOT SIGNED IN ({len(no_shows)})*")
        for i, c in enumerate(no_shows, 1):
            lines.append(f"  {i}. {c['name']} (ID {c['client_id']})\n     Pricing: {c['pricing']}")

    return "\n".join(lines)


def get_multi_noshow_report(class_date=None, class_time=None):
    """Run no-show reports for all classes on a date (optionally filtered by time)."""
    all_classes = _find_all_classes(class_date=class_date, class_time=class_time)
    if not all_classes:
        date_str = class_date or _now().strftime("%Y-%m-%d")
        return [{"error": f"No classes found on {date_str}"}]

    reports = []
    for cls in all_classes:
        report = _get_noshow_report_for(cls)
        reports.append(report)
    return reports


def _get_noshow_report_for(cls):
    """Run no-show report for a pre-resolved class dict."""
    visits = _get_class_visits(cls["id"])

    if not visits:
        return {
            "class_name": cls["name"],
            "class_time": cls["time_fmt"],
            "class_date": cls["date"],
            "instructor": cls["instructor"],
            "total_booked": cls["total_booked"],
            "no_shows": [],
            "signed_in_count": 0,
        }

    no_shows = []
    signed_in_count = 0

    for v in visits:
        if v.get("LateCancelled") or v.get("LateCanceled"):
            continue
        signed_in = v.get("SignedIn", False)
        if signed_in:
            signed_in_count += 1
            continue
        cid = str(v.get("ClientId", ""))
        if not cid:
            continue
        client_name = _lookup_client_name(cid)
        service_obj = v.get("Service") or {}
        service_name = v.get("ServiceName") or service_obj.get("Name") or "N/A"
        no_shows.append({"name": client_name, "client_id": cid, "pricing": service_name})

    return {
        "class_name": cls["name"],
        "class_time": cls["time_fmt"],
        "class_date": cls["date"],
        "instructor": cls["instructor"],
        "total_booked": cls["total_booked"],
        "no_shows": no_shows,
        "signed_in_count": signed_in_count,
    }


def format_multi_noshow_report(reports):
    """Format multiple no-show reports into one WhatsApp message."""
    if len(reports) == 1 and "error" in reports[0]:
        return reports[0]["error"]

    parts = []
    for report in reports:
        parts.append(format_noshow_report(report))

    return "\n\n---\n\n".join(parts)
