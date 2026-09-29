"""One-off: create 'To Do List' and 'Done' lists on the HIIT Office board."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from trello_helper import _trello_params, TRELLO_BASE

HIIT_OFFICE_BOARD_ID = "61e3b3571663315f05905daa"

for name in ["To Do List", "Done"]:
    resp = requests.post(
        f"{TRELLO_BASE}/lists",
        params=_trello_params({"name": name, "idBoard": HIIT_OFFICE_BOARD_ID, "pos": "top"}),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"Created: {data['name']}  (id={data['id']})")
