"""One-off: list Trello boards and their lists so we can set per-user defaults.

Run with:  railway run -- python3 docs/list_trello_boards.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trello_helper import _trello_get


def main():
    boards = _trello_get("members/me/boards", {"fields": "name,id,closed"})
    print(f"Total boards: {len(boards)}\n")
    for b in boards:
        if b.get("closed"):
            continue
        print(f"Board: {b['name']}  (id={b['id']})")
        try:
            lists = _trello_get(f"boards/{b['id']}/lists", {"fields": "name,id,closed"})
            for l in lists:
                if l.get("closed"):
                    continue
                print(f"   - {l['name']}  (id={l['id']})")
        except Exception as e:
            print(f"   (failed to fetch lists: {e})")
        print()


if __name__ == "__main__":
    main()
