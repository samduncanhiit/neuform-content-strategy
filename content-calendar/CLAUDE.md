# Neuform Content Calendar

## What this project is

A 30-day social media content calendar for Chontel Duncan's Neuform fitness app brand. Static HTML deployed to GitHub Pages. Covers Instagram, TikTok, and Stories with full captions, voiceover scripts, and execution notes for each post.

## Deployment

- **Platform:** GitHub Pages
- **Repo:** `samduncanhiit/neuform-content-strategy`
- **Live URL:** https://samduncanhiit.github.io/neuform-content-strategy/
- **The live file is `index.html` in the remote repo** — it must be updated separately via the GitHub API after pushing `neuform-content-calendar.html`
- Deploy workflow: edit `neuform-content-calendar.html` locally, push to remote, then update `index.html` via `gh api`

## Key files

| File | Purpose |
|------|---------|
| `neuform-content-calendar.html` | The main calendar — all 30 days with interactive features |
| `calendar-days4-30.html` | Earlier version / reference (days 4-30 only) |
| `generate_calendar.py` | Script that originally generated the HTML from CSV data |
| `neuform-april-2026-content-calendar.csv` | Source data for the calendar content |
| `neuform-content-strategy.md` | Content strategy document (pillars, goals, brand voice) |
| `build_drive_map.py` | Builds Google Drive folder mapping for upload buttons |
| `drive_folder_map.json` | Cached folder IDs for the upload feature |
| `viral-fitness-videos.json` | Reference data for viral fitness content |
| `NEUFORM_BRAND-GUIDELINES.pdf` | Brand guidelines (colors, tone, visual style) |
| `NEUFORM-STRATEGY-REVIEW.md` | Strategy review notes |

## Calendar features

- **Collapsible days** — click day header to expand/collapse
- **Collapsible posts** — each video (A, B, C) folds to just title + tags
- **Collapsible sections** — Draft Caption, Voiceover Script, Execution Notes each toggle open/closed
- **Edit/Copy buttons** — captions and scripts are editable with localStorage persistence (user must click Save before navigating away — no auto-save)
- **Upload buttons** — Raw/Edited/Approved upload to Google Drive via Railway API
- **Upload tick checkboxes** — track upload status per stage
- **Posted/Scheduled checkboxes** — per-post (A/B/C) scheduling status in day header
- **Filmed checkboxes** — per-post filming status
- **Option swap** — click alternative concepts to switch the active title
- **Mobile hamburger nav** — floating button opens day picker overlay
- **Shot list** — printable film day checklist for upcoming unfilmed days
- **All state persists in localStorage**

## Upload API

The upload buttons hit the Railway-hosted HIIT bot API:
- `POST /api/drive/upload` — upload files to Google Drive
- `GET /api/drive/folders` — get folder map for the calendar

## Dates

Currently runs **April 20 - May 19, 2026** (shifted from original April 1-30).

## Coding conventions

- Single self-contained HTML file (CSS + JS inline)
- All interactivity is vanilla JS (no frameworks)
- State stored in localStorage with separate keys per feature
- DOM manipulation in `<script>` block at end of file
- Brand colors: `--papaya: #FF8F71`, `--boxer-black: #0C0C0C`, `--barbell-grey: #333333`
