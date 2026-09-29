# Claude Code Prompt — Outlook Inbox Email Organiser

Copy and paste this into Claude Code:

---

```
I need you to organise my entire Outlook inbox. Connect to my Outlook account using the Microsoft Graph / Outlook MCP and do the following:

## Step 1: Read all emails in my Inbox
- Fetch all emails from my Inbox (paginate through all of them, not just the first page)
- For each email, note the sender name, sender email address, and subject line

## Step 2: Categorise every email
Go through each email and classify it into one of two categories:

**Advertising/Marketing emails** — These include:
- Newsletters, promotional offers, sales, discount codes
- Marketing emails from brands, retailers, subscription services
- Automated "no-reply" emails that are clearly promotional
- Social media notifications (Facebook, Instagram, LinkedIn, etc.)
- App notifications and product update announcements
- Any email that looks like a mass-send or bulk marketing email

**Regular emails** — Everything else (personal, business, transactional, invoices, etc.)

## Step 3: Create folders and move emails

**For advertising/marketing emails:**
- Create a single folder called "Advertising & Marketing"
- Move ALL advertising/marketing emails into this folder

**For regular emails:**
- Group emails by sender (use the sender's name or company name, not the email address)
- Create a folder for each sender, named cleanly and professionally. Examples:
  - "John Smith" for a person
  - "MindBody" for emails from mindbody
  - "Twilio" for emails from twilio
  - "HIIT Station" for internal business emails
  - "ANZ Bank" for banking emails
  - Use common sense — if multiple email addresses clearly belong to the same company/person, group them together
- Move all emails from each sender into their respective folder

## Step 4: Summary
When done, give me a summary:
- Total emails processed
- How many went into "Advertising & Marketing"
- A list of all folders created with the count of emails in each
- Flag any emails you weren't sure about

## Rules
- Do NOT delete any emails
- Do NOT mark anything as read/unread
- Create all folders as subfolders inside the Inbox (not at the top level)
- If a folder already exists, use it — don't create duplicates
- Process ALL emails, not just recent ones
- Ask me before proceeding if you're unsure about anything
```

---
