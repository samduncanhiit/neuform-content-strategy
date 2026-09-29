# HIIT Station Capalaba — Claude Code Go-Live Prompts

Copy and paste these prompts into Claude Code (VS Code) to complete each task.

---

## 1. Fix the WhatsApp Bot (Restore Full Code + Separate Cron)

### 1a. Create a separate Railway service for the morning briefing cron

```
The cron schedule was removed from the hiit-automations Railway service because it was
preventing the WhatsApp bot web server from running 24/7. I need you to create a NEW
Railway service in the same project (adequate-playfulness) specifically for the morning
briefing cron job.

- Create a new file called briefing_runner.py that imports and runs the briefing from
  mindbody_briefing.py
- It should be deployed as a separate Railway service with cron schedule: 55 20 * * 1-5
  (that's 6:55am Brisbane time weekdays)
- It needs the same env vars: MINDBODY_API_KEY, MINDBODY_SITE_ID, MINDBODY_SOURCE_NAME,
  MINDBODY_SOURCE_PASSWORD, TRELLO_API_KEY, TRELLO_TOKEN, SMTP_HOST, SMTP_PORT, SMTP_USER,
  SMTP_PASS, BRIEFING_RECIPIENT
- Keep the WhatsApp bot (app.py) as the always-on web service
```

---

## 2. Replace Exposed Outlook Client Secret

```
The Outlook Client Secret for the Azure app "Claude MCP Outlook" was accidentally exposed
and needs to be rotated. Here are the details:

- App: Claude MCP Outlook
- Application (client) ID: 131b980b-3537-4471-bb49-38a5f7f96b0f
- Directory (tenant) ID: 39cc7566-e080-4ade-9907-1495cb54c885

Please walk me through:
1. Creating a new client secret in Azure Portal (portal.azure.com > App registrations)
2. Updating the MS_CLIENT_SECRET in Railway environment variables
3. Updating the ms-365-mcp-server config in Claude Code
4. Deleting the old exposed secret from Azure

Give me step-by-step instructions since some of this needs to be done in the browser.
```

---

## 3. Request MindBody Live API Access

```
I have a MindBody developer account with sandbox access. I need to request live API access.

Current details:
- Source name: HIITStationCapalaba
- API Key: <MINDBODY_API_KEY>
- Sandbox Studio ID: -99

Walk me through requesting live access from the MindBody developer portal so I can
connect to our real HIIT Station Capalaba data instead of sandbox data. What information
will they need from me?
```

---

## 4. Build the Combined Daily Briefing

```
Build me a combined daily briefing script that pulls data from ALL connected sources and
emails it to me each morning. It should include:

1. MindBody: Today's class schedule, bookings, active client count, any payment failures
2. Trello: Cards due today or overdue from both HIIT Challenge and HIIT Office boards
3. Google Calendar: Today's events from hiitstationmealplan@gmail.com
4. Outlook: Unread emails or flagged items from the past 24 hours

Format it as a clean HTML email with sections for each source. Send it to the
BRIEFING_RECIPIENT env var. Use the existing MCP connections for Trello, Google Calendar,
and Outlook. Deploy it to the separate briefing cron service on Railway.
```

---

## 5. Get SMTP Credentials for sam@hiitaustralia.com.au

```
I need to set up SMTP sending from sam@hiitaustralia.com.au for the briefing emails.
Help me figure out what SMTP credentials I need from my web host. What information should
I ask my hosting provider for? I need: SMTP_HOST, SMTP_PORT, SMTP_USER, and SMTP_PASS.

Once I have the credentials, help me update the Railway environment variables and test
sending an email.
```

---

## 6. Set Up Twilio WhatsApp Business Number

```
I'm currently using the Twilio WhatsApp sandbox (join bank-discussion, +1 415 523 8886)
for testing. I need to set up a proper WhatsApp Business number for production use.

My Twilio Account SID: <TWILIO_ACCOUNT_SID>
My number: +61420233508

Walk me through:
1. Registering for WhatsApp Business API access through Twilio
2. Getting a dedicated Australian WhatsApp number
3. Completing Meta business verification
4. Updating the webhook URL and TWILIO_WHATSAPP_FROM env var on Railway
```

---

## 7. Add Office Manager to Approved Numbers

```
Add the following phone number to the APPROVED_NUMBERS environment variable on Railway
for the hiit-automations service. The new number is: [PASTE NUMBER HERE]

The current approved numbers are: +61420233508
Update it to include both numbers, comma-separated.
```

---

## 8. Draft an Outlook Email

```
Using the Outlook MCP connection, create a draft email:
- To: [EMAIL ADDRESS]
- Subject: [SUBJECT]
- Body: [WHAT YOU WANT TO SAY]

Create it as a draft in my Outlook so I can review before sending.
```

---

## 9. Check Trello Boards

```
Check both my Trello boards (HIIT Challenge and HIIT Office) and give me a summary of:
- Cards due today or overdue
- Cards in progress
- Any cards assigned to me
- Recent activity in the last 24 hours
```

---

## 10. Check Today's Calendar

```
Check my Google Calendar (hiitstationmealplan@gmail.com) and Outlook calendar for today.
Give me a summary of all meetings, appointments, and events for today and tomorrow.
```

---

## 11. Upgrade the WhatsApp Bot (Future)

```
I want to upgrade the WhatsApp bot to handle more tasks. Currently it's just a basic
Claude chat assistant. Add these capabilities:

1. When I say "draft email to [person] about [topic]" — create an Outlook draft via
   the MS 365 API
2. When I say "check trello" — pull a summary from my Trello boards
3. When I say "what's on today" — check Google Calendar and give me my schedule
4. When I say "briefing" — run the daily briefing on demand

The bot is deployed at ~/Desktop/claude-mcp/app.py on Railway (hiit-automations service).
Add the necessary API integrations and update the requirements.txt.
```

---

## Quick Reference

| Service | URL/Location |
|---------|-------------|
| WhatsApp Bot | hiit-automations-production.up.railway.app |
| Railway Project | adequate-playfulness |
| Twilio Sandbox | +1 415 523 8886 (join bank-discussion) |
| Sam's Number | +61420233508 |
| Project Files | ~/Desktop/claude-mcp/ |
| Azure App ID | 131b980b-3537-4471-bb49-38a5f7f96b0f |
| MindBody Source | HIITStationCapalaba |
