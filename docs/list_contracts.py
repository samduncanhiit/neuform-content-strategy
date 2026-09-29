"""One-off: list every contract product at the site so we can spot membership
types that aren't in TRACKED_MEMBERSHIPS yet.

Run with:  railway run python docs/list_contracts.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindbody_helper import _api_get, TRACKED_MEMBERSHIPS, _is_tracked_membership

def main():
    data = _api_get("sale/contracts", {"LocationId": 1, "limit": 200})
    contracts = data.get("Contracts", []) or []
    print(f"Total contracts returned: {len(contracts)}\n")

    tracked = []
    untracked = []
    for c in contracts:
        name = c.get("Name") or ""
        intro_offer = c.get("IntroOffer")
        active = c.get("ActiveAtSite") or c.get("Active")
        row = (name, intro_offer, active)
        if _is_tracked_membership(name):
            tracked.append(row)
        else:
            untracked.append(row)

    print("=== TRACKED (already in TRACKED_MEMBERSHIPS) ===")
    for n, intro, active in sorted(tracked):
        print(f"  {n}  [intro={intro}, active={active}]")

    print("\n=== NOT TRACKED (potential additions) ===")
    for n, intro, active in sorted(untracked):
        print(f"  {n}  [intro={intro}, active={active}]")

if __name__ == "__main__":
    main()
