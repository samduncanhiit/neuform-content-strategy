# Trello Card Management for HIIT Bot

## Purpose

Let Erin manage her Trello workflow through WhatsApp — add new cards, edit existing ones, mark them done (move to "Done"), and archive mistakes. Today the bot is read-only on Trello (`get_trello_tasks`). This spec adds the write operations.

## Scope

- **Users:** Erin only (`+61421188443`). Sam and Chonnie keep their existing read-only access.
- **Board:** HIIT Office (id `61e3b3571663315f05905daa`).
- **Lists:** Two new lists were pre-created for this workflow: **"To Do List"** (default for new cards) and **"Done"** (where completed cards go).
- **Operations:** add, edit, move, archive.
- **Expansion path:** When Sam/Chonnie's config is provided, only the `USER_TRELLO` dict needs to change; tool logic is user-agnostic.

## Configuration (in `app.py`)

```python
USER_TRELLO = {
    "+61421188443": {
        "board": "HIIT Office",
        "todo_list": "To Do List",
        "done_list": "Done",
    },
}
```

`USER_TRELLO` is the single source of truth for per-user Trello access. Users not in this dict do not see the write tools.

## Tools

All four tools receive the user's phone number (already threaded through `handle_tool_call` via `user_email`; will add `sender` propagation). The board and default lists come from `USER_TRELLO`; the user never names a board.

### 1. `add_trello_card`

Creates a card on the user's board.

| Field | Required | Notes |
|---|---|---|
| `title` | yes | Card name |
| `due_date` | no | `YYYY-MM-DD` (same convention as other date-taking tools in this bot). The tool converts it to the ISO 8601 format Trello expects (`YYYY-MM-DDT10:00:00.000Z`, 10:00 AEST = midnight UTC-ish; fine for a due-date-only workflow). The model gets instructions in the schema to always pass YYYY-MM-DD. |
| `description` | no | Free text |
| `labels` | no | Array of label names; created if missing on the board |
| `list_name` | no | Defaults to `todo_list` ("To Do List"); if provided, fuzzy-matched against the user's board lists |

Response: `"Added '<title>' to <list> on <board>."` plus the Trello short URL.

### 2. `edit_trello_card`

Finds a card by fuzzy title match on the user's board, updates any subset of fields.

| Field | Required | Notes |
|---|---|---|
| `title` | yes | Fuzzy match target — the current title of the card to edit |
| `new_title` | no | Rename |
| `due_date` | no | |
| `description` | no | |
| `labels` | no | Replaces existing labels |

Match logic: case-insensitive substring, then token-overlap score. If exactly one card matches, the tool updates it and returns a confirmation. If multiple cards match, the tool returns a numbered list and asks the user to clarify by including more of the title (no numeric reply flow — keeps the interface stateless).

### 3. `move_trello_card`

Moves a card to the user's Done list (or a named list). This is the "tick off" / "mark done" action.

| Field | Required | Notes |
|---|---|---|
| `title` | yes | Fuzzy match target |
| `list_name` | no | Defaults to `done_list` ("Done") |

Same fuzzy-match + disambiguation rules as `edit_trello_card`.

### 4. `remove_trello_card`

Archives a card. Trello has no hard delete — archive is the standard pattern.

| Field | Required | Notes |
|---|---|---|
| `title` | yes | Fuzzy match target |
| `confirmed` | no | Boolean; defaults to `false` |

**Confirmation flow:**

1. First call (`confirmed=false` or omitted): tool finds the card, returns *"Found '<title>' in <list>. Reply 'yes' to archive."* — does NOT archive.
2. User replies "yes" (or similar affirmative).
3. The model re-calls the tool with `confirmed=true` — the tool archives and returns *"Archived '<title>'."*

A system-prompt rule enforces the two-step flow: the model must never call `remove_trello_card` with `confirmed=true` unless the user has explicitly confirmed in the conversation.

If fuzzy match returns multiple candidates, the first step returns the list and requires a more specific title — same as edit/move.

## `trello_helper.py` changes

Add to the existing helper module:

- `_trello_post(path, params)` and `_trello_put(path, params)` — mirror `_trello_get`.
- `_find_board_id(board_name)` — cached per name.
- `_find_list(board_id, list_name)` — fuzzy match; returns `(list_id, exact_name)` or `None`.
- `_find_card(board_id, title)` — searches all open lists on the board, returns list of match dicts sorted by score.
- `_resolve_labels(board_id, label_names)` — creates missing labels on the board, returns label IDs.
- `create_card(board_id, list_id, title, due_date=None, description=None, label_ids=None)`
- `update_card(card_id, **fields)` — any subset of title/due/description/labels.
- `move_card(card_id, dest_list_id)`
- `archive_card(card_id)`

Existing `_find_hiit_challenge_board` stays unchanged; it's used by the read-only tool.

## `app.py` changes

- Add `USER_TRELLO` dict.
- Add a new `_TRELLO_WRITE_TOOLS` list with the 4 tool definitions.
- In `_get_tools_for_user`, include `_TRELLO_WRITE_TOOLS` only if the caller's phone has an entry in `USER_TRELLO`. (`get_trello_tasks` stays available to all users via `_MINDBODY_TOOLS` or wherever it currently lives — no change to its availability.)
- Thread the caller's phone number (`raw_number`) into `handle_tool_call` so the Trello write tools can look up config.
- Add handler branches in `handle_tool_call` for each of the 4 new tools.
- Update `SYSTEM_PROMPT`:
  - Mention the new Trello write capabilities.
  - Specify the confirmation flow for `remove_trello_card`.
  - Specify that when the user says "I've done X" / "completed X", the model should use `move_trello_card` (to the default done list), not `remove_trello_card`.

## Error handling

- Unknown user (no `USER_TRELLO` entry) trying to call a write tool: tool returns `"Trello write access is not configured for you."` (defense-in-depth; should not happen because tools are filtered out at the schema level).
- List not found after fuzzy match: tool returns `"No list matching '<name>' on <board>. Available lists: ..."`.
- Card not found: tool returns `"No card matching '<title>' on <board>."`.
- Trello API error: log and return `"Trello error: <status>. Try again in a moment."`.

## Caching

- `_find_board_id` result cached in-process indefinitely (boards rarely renamed).
- List lookups cached per board for 10 minutes (lists can be created/renamed).
- No card-level caching — always fresh.

## Out of scope

- Checklists inside cards
- Card comments
- Attachments
- Label color management (labels get default color when auto-created)
- Multi-user board mappings for Sam/Chonnie (awaiting user input)
