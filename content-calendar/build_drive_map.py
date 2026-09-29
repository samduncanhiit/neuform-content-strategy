import warnings
warnings.filterwarnings("ignore")

import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

SERVICE_ACCOUNT_FILE = "hiit-bot-944b90337586.json"
PARENT_FOLDER_ID = "1m99uefGSWlDzT4XZvUk9zlLbb97fsK6T"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
drive = build("drive", "v3", credentials=creds)


def list_folders(parent_id):
    """List all child folders of a parent, handling pagination."""
    folders = []
    page_token = None
    while True:
        resp = drive.files().list(
            q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="nextPageToken, files(id, name)",
            pageSize=100,
            pageToken=page_token,
            orderBy="name",
        ).execute()
        folders.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return folders


result = {}

# 1. Get day folders
day_folders = list_folders(PARENT_FOLDER_ID)
print(f"Found {len(day_folders)} day folders")

for day in sorted(day_folders, key=lambda f: f["name"]):
    day_name = day["name"]
    print(f"  {day_name}")
    day_entry = {"id": day["id"], "posts": {}}

    # 2. Get post folders inside each day
    post_folders = list_folders(day["id"])
    for post in sorted(post_folders, key=lambda f: f["name"]):
        post_name = post["name"]
        post_num = post_name.split(" ")[0]  # e.g. "01"
        print(f"    {post_name}")
        post_entry = {"id": post["id"], "name": post_name}

        # 3. Get Raw/Edited/Approved subfolders
        sub_folders = list_folders(post["id"])
        for sub in sub_folders:
            if sub["name"] in ("Raw", "Edited", "Approved"):
                post_entry[sub["name"]] = sub["id"]

        day_entry["posts"][post_num] = post_entry

    result[day_name] = day_entry

out_path = "drive_folder_map.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved to {out_path}")
print(f"Total days: {len(result)}")
total_posts = sum(len(d["posts"]) for d in result.values())
print(f"Total posts: {total_posts}")
