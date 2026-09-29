#!/usr/bin/env python3
"""Generate HIIT Operations Manual as a Word document."""

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
import datetime

doc = Document()

# --- Styles ---
style = doc.styles['Normal']
font = style.font
font.name = 'Calibri'
font.size = Pt(11)

for level in range(1, 4):
    hs = doc.styles[f'Heading {level}']
    hs.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)

# --- Title Page ---
doc.add_paragraph()
doc.add_paragraph()
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('HIIT STATION CAPALABA')
run.bold = True
run.font.size = Pt(28)
run.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('Operations Manual')
run.bold = True
run.font.size = Pt(22)

info = doc.add_paragraph()
info.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = info.add_run(f'\nHIIT Australia Pty Ltd\nABN: 63 167 814 682\n\nGenerated: {datetime.date.today().strftime("%d %B %Y")}')
run.font.size = Pt(12)
run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

doc.add_page_break()

# --- Table of Contents placeholder ---
doc.add_heading('Table of Contents', level=1)
toc_items = [
    '1. Business Overview',
    '2. Calendar & Schedule',
    '3. Office Manager Daily Tasks',
    '4. Day-of-Week Specific Duties',
    '5. Office Manager Monthly Tasks',
    '6. Office Support Monthly Tasks',
    '7. Membership Management',
    '8. Financial Processes',
    '9. Lead Management & Follow-Up',
    '10. Reception Duties & Instructions',
    '11. Staff Management & Rostering',
    '12. Events & Friday Takeovers',
    '13. 8-Week Challenge Operations',
    '14. Content & Marketing',
    '15. Revenue Generation Ideas',
    '16. Facility & Equipment Management',
    '17. General Reference Information',
    '18. Non-Priority / Backlog Actions',
    '19. Completed Items Archive (2025-2026)',
    '20. Automation & Integration Recommendations',
]
for item in toc_items:
    p = doc.add_paragraph(item)
    p.paragraph_format.space_after = Pt(2)

doc.add_page_break()

# ============================================================
# SECTION 1 - BUSINESS OVERVIEW
# ============================================================
doc.add_heading('1. Business Overview', level=1)
doc.add_paragraph(
    'HIIT Station Capalaba is a fitness facility offering group HIIT (High-Intensity Interval Training) classes, '
    'strength training, Muay Thai, open gym access, a PT studio, sauna, and an 8-week body transformation challenge program. '
    'The business operates under HIIT Australia Pty Ltd (ABN 63 167 814 682).'
)
doc.add_paragraph(
    'Operations are managed through two primary Trello boards: "HIIT Office" for day-to-day office and gym management, '
    'and "HIIT Challenge" for running the recurring 8-week challenge programs. '
    'The Google Calendar (hiitstationmealplan@gmail.com) is used for scheduling meetings, challenge events, '
    'public holiday closures, and content deadlines.'
)
doc.add_heading('Key Systems & Software', level=2)
systems = [
    ('MindBody (MB)', 'Core membership, scheduling, reporting, and point-of-sale system (Site ID: 5728999)'),
    ('Trello', 'Task management for office operations, challenges, and maintenance'),
    ('Deputy', 'Staff rostering and shift management'),
    ('Canva', 'Design tool for social media posts, public holiday closures, challenge graphics'),
    ('Google Calendar', 'Scheduling for team meetings, management meetings, challenge events, public holiday closures, and content deadlines (hiitstationmealplan@gmail.com)'),
    ('Google Drive', 'Shared documents including spreadsheets for cancellations, suspensions, debt collection, equipment, incident reports'),
    ('HelloSign', 'Digital contract signing for memberships'),
    ('Stripe', 'Online/QR code payment processing'),
    ('SMS Broadcast', 'Bulk SMS communication to members and challengers'),
    ('Calendly', 'Sauna booking system'),
    ('Active Campaign', 'Email marketing automation'),
    ('InBody (LookInBody)', 'Body composition scanning for challenge participants'),
    ('Cronometer', 'Nutrition tracking (used for challenge meal plans)'),
    ('Facebook & Instagram', 'Social media platforms and member forums (HIIT Capalaba forum, HIIT Society on Instagram)'),
    ('Ring Camera', 'Facility security cameras'),
    ('Sonos / Soundcloud / Spotify', 'Music systems for gym floor and PT studio'),
]
table = doc.add_table(rows=1, cols=2)
table.style = 'Light Grid Accent 1'
hdr = table.rows[0].cells
hdr[0].text = 'System'
hdr[1].text = 'Purpose'
for s, p in systems:
    row = table.add_row().cells
    row[0].text = s
    row[1].text = p

doc.add_page_break()

# ============================================================
# SECTION 2 - CALENDAR & SCHEDULE
# ============================================================
doc.add_heading('2. Calendar & Schedule', level=1)
doc.add_paragraph(
    'The HIIT Station business rhythm is driven by a combination of daily operations, weekly team meetings, '
    'recurring management check-ins, and the 8-week challenge cycle. The Google Calendar (hiitstationmealplan@gmail.com) '
    'is the source of truth for scheduling. Below is the full breakdown of recurring and scheduled events.'
)

doc.add_heading('Daily Recurring Operations', level=2)
daily_cal = [
    ('Team Meeting', 'Every weekday, 9:15-9:45am. Attendees: Chontel, Emily, Erin, Sam, admin@hiitaustralia.com.au. This is the core daily sync for all office and coaching staff.'),
    ('Challenge Things (during active challenge rounds)', 'Daily, 11:00am-12:00pm. Check the Challenge Trello board for tasks that need to be completed that day.'),
    ('Check RD Facebook Requests (during active challenge rounds)', 'Daily recurring. Monitor the current challenge round\'s Facebook group for new join requests and approve/action as needed.'),
]
for name, desc in daily_cal:
    p = doc.add_paragraph()
    run = p.add_run(f'{name}: ')
    run.bold = True
    p.add_run(desc)

doc.add_heading('Weekly Meetings', level=2)
doc.add_paragraph(
    'The daily Team Meeting (Mon-Fri, 9:15-9:45am) serves as the weekly rhythm. '
    'Use this meeting to align on daily priorities, review leads, discuss member issues, and coordinate challenge tasks.'
)

doc.add_heading('Management Meetings', level=2)
mgmt_meetings = [
    ('Management Team Meeting @ HIIT', 'Recurring (frequency set in calendar). Attendees: Chontel, Erin, Sam, jtukaokao@yahoo.com (Michelle). Prepare the monthly management report 2 days before this meeting (see Section 5).'),
    ('Quarterly Team Meeting @ HIIT', 'Attendees: Chontel, Erin, Sam, jtukaokao@yahoo.com (Michelle). Next scheduled: 12 May 2026. Use for strategic planning, reviewing challenge results, and setting quarterly goals.'),
]
for name, desc in mgmt_meetings:
    p = doc.add_paragraph()
    run = p.add_run(f'{name}: ')
    run.bold = True
    p.add_run(desc)

doc.add_heading('Public Holiday Closures (Calendar-Confirmed)', level=2)
doc.add_paragraph(
    'The following closures are set as recurring annual events in the Google Calendar. '
    'These are in addition to the full QLD public holiday list (see Section 12).'
)
cal_closures = [
    'HIIT CLOSED - 1 January (New Year\'s Day) - recurring annually',
    'HIIT CLOSED - 26 January (Australia Day) - recurring annually',
]
for c in cal_closures:
    doc.add_paragraph(c, style='List Bullet')
doc.add_paragraph(
    'Note: All QLD public holidays for 2026 have been cancelled in the MindBody schedule. '
    'Closure posts must be created in Canva and scheduled 1 week in advance. '
    'Recurring calendar reminders were set up in March 2026 - verify dates each year.'
)

doc.add_heading('8-Week Challenge Cycle', level=2)
doc.add_paragraph(
    'Challenges run back-to-back throughout the year with a short transition period between rounds. '
    'Each round generates its own set of calendar events covering seminars, admin tasks, and content deadlines. '
    'See Section 13 for the full challenge operations detail including the RD63 and RD64 timelines.'
)

doc.add_heading('Challenge Round Overview (2026)', level=3)
rounds_table = doc.add_table(rows=1, cols=4)
rounds_table.style = 'Light Grid Accent 1'
hdr = rounds_table.rows[0].cells
hdr[0].text = 'Round'
hdr[1].text = 'Status'
hdr[2].text = 'Start Date'
hdr[3].text = 'End Date'
rounds_data = [
    ('RD63', 'Completed', '(~Jan 2026)', '14 Mar 2026'),
    ('RD64', 'Upcoming', '18 Apr 2026', '13 Jun 2026'),
]
for rd, status, start, end in rounds_data:
    row = rounds_table.add_row().cells
    row[0].text = rd
    row[1].text = status
    row[2].text = start
    row[3].text = end

doc.add_heading('Typical Challenge Calendar Pattern', level=3)
doc.add_paragraph('Each 8-week challenge round follows this calendar pattern:')
pattern = [
    'Pre-challenge (3-4 weeks before): Finalise details, create meal plans, set up MB promos, send "week before" info',
    'Week 1: First day of challenge, InBody scans, welcome comms',
    'Week 2-3: How to Macro Seminar, Barbell Fundamentals class',
    'Week 4-5: Midway Seminar with Chontel, midway check-ins',
    'Week 6-7: Post Nutrition Seminar, final push content',
    'Week 8: Final day, end-of-challenge photography, announce BOW winner',
    'Post-challenge (1-2 weeks after): Edit results tiles, remove challengers from Cronometer, deactivate promos, set up next round calendar',
]
for p_item in pattern:
    doc.add_paragraph(p_item, style='List Bullet')

doc.add_heading('Content & Admin Tasks from Calendar', level=2)
doc.add_paragraph('The following recurring content tasks are tracked via calendar:')
cal_content = [
    'Film HALF HALF change video with Sam (with Chontel)',
    'Film Sam - "why our challenge works" content piece',
    'Set up next challenge events and reminders in calendar after each round ends',
    'HIIT Kids Promo scheduling',
    'FROPRO delivery coordination',
]
for c in cal_content:
    doc.add_paragraph(c, style='List Bullet')

doc.add_page_break()

# ============================================================
# SECTION 3 - OFFICE MANAGER DAILY TASKS
# ============================================================
doc.add_heading('3. Office Manager Daily Tasks', level=1)
doc.add_paragraph(
    'The Office Manager is responsible for the following tasks every working day:'
)
daily_tasks = [
    'Attend daily Team Meeting (Mon-Fri, 9:15-9:45am) with Chontel, Emily, Erin, and Sam',
    'Check and respond to all emails (Mac Mail and Gmail - including chontelahu88 and hiitstationmealplans)',
    'Reply to phone queries, phone calls, and social media messages (if no office support on shift)',
    'Check and reconcile cash drawer against MindBody reports',
    'Rerun failed credit card payments via MB Dashboard',
    'Check the Maintenance Trello board and action items as required',
    'Check MindBody leads under Lead Management section',
    'Verify reception staff completed their daily tasks',
    'Action all suspension and cancellation requests (must be in writing via email)',
    'Check the HIIT Challenge Trello board daily',
    'Check HelloSign for any outstanding unsigned contracts - follow up as needed',
    'Check Deputy shifts are entered correctly; adjust and notify Michelle of changes',
    'Follow up on stock discrepancies flagged in reception reports',
]
for t in daily_tasks:
    doc.add_paragraph(t, style='List Bullet')

doc.add_heading('Daily Lead Follow-Up Process', level=2)
doc.add_paragraph(
    'This is critical to client retention and must be done daily (except Saturdays - follow up Monday).'
)
lead_steps = [
    'In MindBody, go to Marketing > Lead Management. Review every new lead and follow up accordingly.',
    'Run "First Class" report: Clients > First Visit > All > Detail. Contact anyone who attended their first class.',
    'Run "Sales By Service" report: Service Category > Intros and Offers > All > Detail.',
    'Send follow-up SMS (templates in HIIT Phone notes) to all first-class attendees.',
    'Check pre-screen forms are completed for all first-time visitors.',
    'Add all details to the Enquiries spreadsheet in Google Drive.',
    'Follow up all enquiries: Membership, Challenge, and First Class.',
    'Work through reports in both AM and PM shifts. MUST BE COMPLETED DAILY.',
]
for i, s in enumerate(lead_steps, 1):
    doc.add_paragraph(f'{i}. {s}')

doc.add_page_break()

# ============================================================
# SECTION 4 - DAY-OF-WEEK SPECIFIC DUTIES
# ============================================================
doc.add_heading('4. Day-of-Week Specific Duties', level=1)

doc.add_heading('Monday', level=2)
monday_tasks = [
    'Check Maintenance Trello board',
    'Staff roster check',
    'Run MindBody reports: Unpaid Visits and Cancellations',
    'Check unpaid visits: Reports > Clients > Unpaid Visits. Investigate and resolve each.',
    'Check cancellations are actioned correctly: cross-reference Google Drive spreadsheet with MindBody (check contract terminated, member status, remove from Facebook forum and Instagram HIIT Society)',
    'Check sign-ups: Reports > Clients > New Members. Add to Trello and spreadsheet, contact new members.',
    'Update the Class Coach Script in Google Drive and print for coaches desk upstairs.',
    'Post Friday Takeover follow-up: email Michelle charity details (charity name, link, total cash, when envelope sent).',
]
for t in monday_tasks:
    doc.add_paragraph(t, style='List Bullet')

doc.add_heading('Tuesday', level=2)
tuesday_tasks = [
    'Review Monday reception stock check list and order anything needed.',
    'All purchases must be approved by Michelle. Physical credit card is at the Warehouse; photocopy in the safe.',
    'Supplements: order directly through Blair at blair@powersupps.com.au.',
    'Flooring: Eclipse Flooring Solutions - cameron@eclipsefloorsolutions.com.au (1 week turnaround).',
    'InBody products: order from inbody.net.au/shop/ - check stock levels during every challenge.',
    'Bins go out on Tuesday night (collected Wednesday morning, brought back Wednesday night).',
]
for t in tuesday_tasks:
    doc.add_paragraph(t, style='List Bullet')

doc.add_heading('Wednesday', level=2)
doc.add_paragraph('No additional specific actions beyond daily tasks.')

doc.add_heading('Thursday', level=2)
thursday_tasks = [
    'PT Studio check: walk through for tidiness and cleanliness. Notify staff. Check machines - tighten screws, nuts, bolts. Check bike foot pedals.',
    'Run Account Arrears report: MindBody > Reports > Clients > Account Balance. Export to Excel, copy to arrears spreadsheet.',
    'IMPORTANT: Filter out challenge participants who owe for upcoming challenges.',
    'Also run failed autopays report from MB Dashboard > Failed Autopays.',
    'Contact each client based on how many times previously notified.',
    'Step-by-step arrears guide available in Google Drive.',
]
for t in thursday_tasks:
    doc.add_paragraph(t, style='List Bullet')

doc.add_heading('Friday', level=2)
friday_tasks = [
    'Re-run overdue accounts:',
    '  - Credit Card: use "Failed Autopays" tab > select payment > "Run all selected transactions now"',
    '  - Direct Debit: Point of Sale > Search Client > Payment/Gift Cards > Account Payment > enter amount > comment "arrears" > Add Item > select DIRECT DEBIT > complete sale (24-48 hrs to clear)',
    'Send weekly wrap-up text to staff group chat including: how the week was, anything for noting, complaints/compliments, important info, upcoming events.',
]
for t in friday_tasks:
    doc.add_paragraph(t, style='List Bullet')

doc.add_page_break()

# ============================================================
# SECTION 5 - OFFICE MANAGER MONTHLY TASKS
# ============================================================
doc.add_heading('5. Office Manager Monthly Tasks', level=1)

doc.add_heading('Monthly Management Report', level=2)
doc.add_paragraph(
    'Prepare 2 days before the recurring Management Team Meeting (see Section 2 for meeting schedule). Include:'
)
report_items = [
    'Month\'s issues: maintenance, stocktake, financial, staff sickness/issues, member issues',
    'Class counts: MindBody > Reports > Clients > Attendance Analysis (sort by HIIT Capalaba, Day of week)',
    'Sauna counts: log into Calendly (calendly.com/event_types/user/me), review previous month bookings',
    'Comparison of cancellations vs first-time visits',
    'Run New Members report (3 & 6 month contracts) every Friday and add to Google Drive',
    'Verify all new members have been contacted',
    'Send cancellation report to Office Group Chat with context on reasons',
    'Calculate total members: Total - staff - challenge-only + declined + suspended',
]
for item in report_items:
    doc.add_paragraph(item, style='List Bullet')

doc.add_heading('Free Memberships Audit', level=2)
doc.add_paragraph('Check monthly. Categories: Staff, Staff partners, Friends & family, Service swap (Sarah Dales only). Spreadsheet in Google Drive.')

doc.add_heading('Debt Collection Review (Every 3 Months)', level=2)
doc.add_paragraph(
    'Check the Debt Collection Spreadsheet and the ARMA Group portal for updates. '
    'Login: cx.armaonline.com.au (User: HIITCAPALABA). '
    'Contact: consumer@armagroup.com.au or David@armagroup.com.au. '
    'If a client\'s debt is cleared, remove MB suspension, terminate membership, and update spreadsheet.'
)

doc.add_heading('Incident Report Review (Last Thursday of Month)', level=2)
doc.add_paragraph(
    'Review all incidents for the month. Follow up on anything required. '
    'Note repeated injuries and develop prevention plans. Spreadsheet in Google Drive.'
)

doc.add_heading('Gym Walk-Through (Last Thursday of Month)', level=2)
doc.add_paragraph('Walk through the gym with Janelle. Document all maintenance needs in the Maintenance Trello board.')

doc.add_page_break()

# ============================================================
# SECTION 6 - OFFICE SUPPORT MONTHLY TASKS
# ============================================================
doc.add_heading('6. Office Support Monthly Tasks', level=1)

doc.add_heading('Lost Property Procedure (Last Monday of Month)', level=2)
lost_property = [
    'Organise lost property bin and photograph everything neatly laid out.',
    'Post in forums: items not collected in ONE WEEK will be donated.',
    'If a client claims an item via text, remove it immediately and label for collection.',
    'After one week, only dispose of items from the original photo (not newer items).',
]
for i, s in enumerate(lost_property, 1):
    doc.add_paragraph(f'{i}. {s}')

doc.add_heading('Sleeper Members Check (Last Friday of Month)', level=2)
doc.add_paragraph(
    'Check all sleeper members have not been attending on a discounted/old membership. '
    'If they have attended, contact about membership price increase.'
)
doc.add_paragraph('Current sleeper list:')
sleepers = [
    'Stephanie Rogers', 'Alexandra Leimgruber', 'Alex Hynes', 'Ashlee McDonnell',
    'Wayne Molander', 'Rhiannon Saltwell', 'Sean Cooke', 'Krystle Van Egmond',
    'Stacey Sobolewski', 'Bianca Andraczke'
]
for s in sleepers:
    doc.add_paragraph(s, style='List Bullet')

doc.add_heading('Bins (Weekly - Tuesday Night)', level=2)
doc.add_paragraph(
    'Bins out Tuesday night, collected Wednesday morning, brought back Wednesday night. '
    'Green bin collected every second Tuesday via JJ Richards. Monthly invoice sent to Michelle for payment.'
)

doc.add_heading('Check Links & Forms (Last Thursday of Month)', level=2)
doc.add_paragraph('Verify all website links, forms, and booking links are working correctly.')

doc.add_page_break()

# ============================================================
# SECTION 7 - MEMBERSHIP MANAGEMENT
# ============================================================
doc.add_heading('7. Membership Management', level=1)

doc.add_heading('2026 Membership Tracking', level=2)
doc.add_paragraph(
    'Monthly sign-ups and cancellations are tracked on the HIIT Office Trello board under "2026 MEMBERSHIPS". '
    'Each month has a cancellations card and a sign-ups card with member names, dates, and contract types.'
)

doc.add_heading('Cancellation Process', level=2)
cancel_steps = [
    'Client must submit cancellation request via email (written confirmation required).',
    'Action in MindBody: Search Client > Account Details > verify no active green membership > check contract is red/terminated.',
    'Select More Info > confirm Member Status shows "Terminated".',
    'Remove from Facebook HIIT Forum (search name, select Remove).',
    'Remove from Instagram HIIT Society (search followers, remove follower).',
    'Update Google Drive spreadsheet: highlight green, add checked date, tick removed from FB and Insta.',
]
for i, s in enumerate(cancel_steps, 1):
    doc.add_paragraph(f'{i}. {s}')

doc.add_heading('Suspension Process', level=2)
doc.add_paragraph('All suspension requests must be in writing via email. Action daily alongside cancellations.')

doc.add_heading('Student Memberships (Yearly Review)', level=2)
doc.add_paragraph('End of year (November): run MindBody report for student memberships, check expiries, deactivate school leavers.')
doc.add_paragraph('Start of year (February): collect new student IDs, save in Google Drive, flag expiry dates on MindBody accounts.')
doc.add_paragraph('Current students: Hamish Marty (expires Mar 2026), Hannah Bellamy (Dec 2026), Ava Davies (Dec 2026).')

doc.add_heading('Emergency Services Memberships', level=2)
es_steps = [
    'Client must be verified emergency services personnel.',
    'Must email requesting the membership.',
    'ID must be presented and saved to Google Drive.',
    'Add member to Google spreadsheet.',
    'Add staff alert in MB when ID expires.',
    'Add membership via Point of Sale, send contract for signing.',
    'Check IDs every 3 months to ensure they are still current.',
]
for i, s in enumerate(es_steps, 1):
    doc.add_paragraph(f'{i}. {s}')

doc.add_heading('Sheldon College Parents - Free 2 Weeks', level=2)
doc.add_paragraph(
    'Sheldon College parents are offered 2 free weeks. Enquiries come via email or phone. '
    'Activate via Point of Sale under "Sheldon College Parents 2 weeks free". '
    'Also add FREE 24/7 membership (14 days). Follow up regularly. Notify coaches when they attend classes.'
)

doc.add_heading('Age Restrictions', level=2)
age_rules = [
    'Under 14: NOT permitted to train.',
    '14 years: Muay Thai ONLY, with parent/guardian permission AND supervision.',
    '15 years: Strength training and Muay Thai, with parent/guardian permission AND supervision.',
    '16-17 years: No parent permission needed to train, BUT cannot sign contracts without parent/guardian. Parent must complete debit success form in their own name.',
    '18+: Can sign contracts and hold memberships independently.',
]
for r in age_rules:
    doc.add_paragraph(r, style='List Bullet')

doc.add_page_break()

# ============================================================
# SECTION 8 - FINANCIAL PROCESSES
# ============================================================
doc.add_heading('8. Financial Processes', level=1)

doc.add_heading('Cash Drawer Reconciliation (Daily)', level=2)
cash_steps = [
    'Run MindBody report: Reports > Sales > Cash Drawer > select date > HIIT Station Capalaba > GO.',
    'Cross-check EFT & Cash totals against the envelope contents AND back-end payments (no receipt for these).',
    'If discrepancy found: investigate, note on envelope, escalate to Michelle if significant.',
    'Give envelopes to Allexis weekly on Mondays for Michelle at the Warehouse.',
    'Maintain sufficient notes and coins in safe - request change from Michelle if needed.',
]
for i, s in enumerate(cash_steps, 1):
    doc.add_paragraph(f'{i}. {s}')

doc.add_heading('Stripe Payments (After Hours)', level=2)
doc.add_paragraph(
    'Check Stripe (dashboard.stripe.com) for any after-hours QR code purchases. '
    'Login > Transactions > check current date > complete a Stripe receipt (date, client, stock, amount) > '
    'process in MB as "STRIPE PAYMENT" > put receipt in till for end-of-day matching.'
)

doc.add_heading('Account Arrears Process', level=2)
doc.add_paragraph('Run every Friday:')
arrears_steps = [
    'MindBody > Reports > Clients > Account Balance > generate report > export to Excel > copy to arrears spreadsheet.',
    'Also run: MB Dashboard > Failed Autopays > generate > export > copy to spreadsheet.',
    'Filter out challenge participants who owe for upcoming challenges.',
    'Contact clients based on escalation level (1st, 2nd, 3rd attempt).',
]
for i, s in enumerate(arrears_steps, 1):
    doc.add_paragraph(f'{i}. {s}')

doc.add_heading('Debt Collection Process', level=2)
doc.add_paragraph('After 3 failed attempts to recover debt:')
debt_steps = [
    'Add client details to Debt Collection spreadsheet.',
    'Email spreadsheet to consumer@armagroup.com.au.',
    'Place open-ended suspension on MindBody account (no further payments, no class bookings).',
    'Wipe debt from MindBody.',
    'Add alert flag on MB account regarding debt collection.',
    'Update Debt Collection spreadsheet.',
    'Notify Office Chat about new debt collection clients.',
    'Check ARMA portal every 3 months for updates.',
    'Once debt cleared: remove alert, release suspension, terminate membership.',
]
for i, s in enumerate(debt_steps, 1):
    doc.add_paragraph(f'{i}. {s}')

doc.add_heading('Ordering & Purchasing', level=2)
doc.add_paragraph(
    'All purchases require Michelle\'s approval. Physical credit card at Warehouse; '
    'photocopy in safe for online purchases. If physical card needed, coordinate with Michelle for pickup/return.'
)
doc.add_paragraph('Key suppliers:')
suppliers = [
    ('Supplements & Cans', 'Blair - blair@powersupps.com.au (direct order via email)'),
    ('Flooring', 'Eclipse Flooring Solutions - cameron@eclipsefloorsolutions.com.au (1 week turnaround)'),
    ('InBody Products', 'inbody.net.au/shop/ (check stock during every challenge)'),
    ('Towels', 'Ordered through Michelle via The Move Better Project. Order 10 at a time when stock reaches 10.'),
    ('Waste Collection', 'JJ Richards (green bin, every second Tuesday). Monthly invoice to Michelle.'),
]
for name, detail in suppliers:
    p = doc.add_paragraph()
    run = p.add_run(f'{name}: ')
    run.bold = True
    p.add_run(detail)

doc.add_page_break()

# ============================================================
# SECTION 9 - LEAD MANAGEMENT
# ============================================================
doc.add_heading('9. Lead Management & Follow-Up', level=1)
doc.add_paragraph(
    'Lead follow-up is essential to client retention. See Section 3 for the daily lead process. '
    'All JotForm enquiry details must be added to the Enquiry Follow-Up spreadsheet. '
    'These leads may convert to memberships.'
)
doc.add_paragraph(
    'Ensure all enquiries (Membership, Challenge, First Class) are followed up. '
    'Reception team can help make phone calls during their shifts. '
    'Sort spreadsheet and distribute follow-up tasks.'
)

doc.add_page_break()

# ============================================================
# SECTION 10 - RECEPTION DUTIES
# ============================================================
doc.add_heading('10. Reception Duties & Instructions', level=1)

doc.add_heading('Daily Reception Tasks', level=2)
reception_daily = [
    'Complete the cleaning checklist every shift (daily tasks + contribute to weekly tasks).',
    'Restock all areas: cans, paper towels (upstairs and downstairs), fridges, toiletries.',
    'Check and empty bins or push down as needed.',
    'Be active on all forums - especially during Challenge. Interact with every client post.',
    'Respond to socials daily: FB Messenger, Instagram, Meta, FB forums. Reshare posts HIIT is tagged in.',
    'Close till if no PM reception is on. Lock up, give envelope to Office Manager.',
    'Check Stripe for after-hours payments.',
]
for t in reception_daily:
    doc.add_paragraph(t, style='List Bullet')

doc.add_heading('New Staff Onboarding Checklist', level=2)
new_staff = [
    'Uniform and badge - explain what\'s off limits (random jumpers/colours)',
    'Presentation standards',
    'Deputy setup and how to log shifts',
    'Full facility walkthrough - where everything is',
    'HIIT phone orientation',
    'Reception role overview and what the position means to the company',
    'Childminding requirements and responsibilities with parents',
    'Socials training: editing, CapCut, Google Drive',
    'Use of facility as staff - who you represent',
    'First aid overview (longer session with all staff at later date)',
    'Set up free membership in MindBody',
    'Set up staff profile in MindBody',
]
for i, s in enumerate(new_staff, 1):
    doc.add_paragraph(f'{i}. {s}')

doc.add_heading('Reception Staffing Guidelines', level=2)
doc.add_paragraph('Employee Type: Casual employees classified at Level 2.')
guidelines = [
    'Minimum engagement: 1 hour per shift.',
    'Maximum ordinary hours: 10 hours per day.',
    'Monday-Friday: ordinary hours 5:00am - 11:00pm.',
    'Saturday-Sunday: ordinary hours 6:00am - 9:00pm.',
    'Work before these start times = overtime.',
    'Split shifts allowed (max 2 parts), cannot exceed max hours per day.',
    '30-minute break required for shifts over 5 hours.',
    'Reception cannot work 3 broken shifts.',
    '10-hour break required between finishing at night and morning shift.',
]
for g in guidelines:
    doc.add_paragraph(g, style='List Bullet')

doc.add_page_break()

# ============================================================
# SECTION 11 - STAFF MANAGEMENT
# ============================================================
doc.add_heading('11. Staff Management & Rostering', level=1)
doc.add_paragraph(
    'All rostering and roster changes are managed through Deputy. The Office Manager is responsible for: '
    'roster changes, monthly rosters, communicating changes in Office group, filling vacant shifts.'
)
doc.add_paragraph('Maintain at least 2 weeks of rosters in advance. Confirm any shift requirements with Michelle.')
doc.add_paragraph('See Google Drive for staffing info and award details.')

doc.add_page_break()

# ============================================================
# SECTION 12 - EVENTS & FRIDAY TAKEOVERS
# ============================================================
doc.add_heading('12. Events & Friday Takeovers', level=1)

doc.add_heading('Friday Takeover Events', level=2)
doc.add_paragraph('Monthly themed events with DJ, BBQ, charity fundraising, and "Bring a Friend for Free".')

doc.add_heading('2026 Friday Takeover Schedule', level=2)
takeover_dates = [
    ('January 16', 'Charity (Dalton Walsh and Family)'),
    ('February 13', 'Valentine\'s Day theme (Charity: Katrina Oakley/Reece - stage 3 cervical cancer)'),
    ('March 20', 'Charity (continuing Feb charity)'),
    ('April 24', 'ANZAC theme'),
    ('May 8', 'Mother\'s Day theme'),
    ('June 19', 'Charity'),
    ('July 24', 'Christmas in July theme'),
    ('August 14', 'HIIT Birthday theme'),
    ('September 4', 'Father\'s Day theme'),
    ('October 30', 'Halloween theme'),
    ('November 13', 'Charity'),
]
table2 = doc.add_table(rows=1, cols=2)
table2.style = 'Light Grid Accent 1'
hdr = table2.rows[0].cells
hdr[0].text = 'Date'
hdr[1].text = 'Theme'
for d, t in takeover_dates:
    row = table2.add_row().cells
    row[0].text = d
    row[1].text = t

doc.add_heading('Takeover Checklist', level=2)
doc.add_paragraph('Shopping list: 4x white bread, 2x bulk sausage packs, 1x bag onions, 4x BBQ trays, sauces, napkins.')
doc.add_paragraph('Post on HIIT Society. Ensure better childminding process - second booking system for independent kids zone.')
doc.add_paragraph('Sam\'s equipment needs: BBQ, high black tables, gas bottle, esky fridge, DJ equipment.')

doc.add_heading('Free Takeover Membership', level=2)
doc.add_paragraph(
    'A free membership is awarded monthly to a Takeover attendee. Winners receive approximately 4 weeks of free membership.'
)
doc.add_paragraph('2026 Winners so far:')
winners = [
    'January: Chris Donnelly (19/01/26 - 16/02/26)',
    'February: Lewis Gettons (16/02/26 - 15/03/26)',
    'March: Jules Trethaway (23/03/26 - 23/04/26)',
]
for w in winners:
    doc.add_paragraph(w, style='List Bullet')

doc.add_heading('Post-Event Process (Monday)', level=2)
doc.add_paragraph(
    'After any donation/fundraising event, email Michelle with: charity name, charity link (e.g. GoFundMe), '
    'total cash collected, when the cash envelope is being sent.'
)

doc.add_heading('Public Holiday Closures', level=2)
doc.add_paragraph(
    'Create closure posts in Canva and schedule in advance on Instagram and Facebook forums. '
    'Always confirm with Head Office that the gym is closing. Check Canva for existing templates. '
    'New Year\'s Day (1 Jan) and Australia Day (26 Jan) are confirmed as recurring annual closures in the Google Calendar (see Section 2).'
)
doc.add_paragraph('2026 Queensland Public Holidays (all cancelled in MB schedule):')
holidays = [
    "New Year's Day - Thu 1 Jan",
    'Australia Day - Mon 26 Jan',
    'Good Friday - Fri 3 Apr',
    'Easter Saturday - Sat 4 Apr',
    'Easter Sunday - Sun 5 Apr',
    'Easter Monday - Mon 6 Apr',
    'Anzac Day - Sat 25 Apr',
    'Labour Day - Mon 4 May',
    'Royal Queensland Show - Mon 10 Aug (Redlands: Wed 12 Aug)',
    "King's Birthday - Mon 5 Oct",
    'Christmas Eve - Thu 24 Dec (from 6pm)',
    'Christmas Day - Fri 25 Dec',
    'Boxing Day - Sat 26 Dec',
    'Additional Boxing Day - Mon 28 Dec',
]
for h in holidays:
    doc.add_paragraph(h, style='List Bullet')

doc.add_heading('Annual Event Calendar', level=2)
events = [
    "February 14 - Valentine's Day",
    "March 8 - International Women's Day",
    "March 17 - St. Patrick's Day",
    'April - HIIT Games / Easter / Anzac Day (25th)',
    "May - Mother's Day",
    'July - 4x48 event (dates TBC, collab with Wanderlust)',
    'September - R U OK? Day',
    'October - Breast Cancer Awareness Month / Halloween (31st)',
    'November - Movember',
    'December - Christmas appeal for homeless / Christmas / NYE',
]
for e in events:
    doc.add_paragraph(e, style='List Bullet')

doc.add_page_break()

# ============================================================
# SECTION 13 - 8-WEEK CHALLENGE
# ============================================================
doc.add_heading('13. 8-Week Challenge Operations', level=1)
doc.add_paragraph(
    'HIIT Station runs recurring 8-week body transformation challenges. These are managed via the "HIIT Challenge" '
    'Trello board (lists moved to a separate 2026 board in December 2025) and the Google Calendar.'
)

doc.add_heading('Challenge Board Structure', level=2)
challenge_lists = [
    ('CHALLENGE QUICK INFO', 'Key dates, pricing, and challenge overview information'),
    ('NEW WAY TO RUN CHALLENGE', 'Weekly to-do lists for each week of the 8-week challenge (Weeks 1-7 + pre-challenge)'),
    ('SMS COMMUNICATION', 'Pre-written SMS messages to send each week to challengers'),
    ('NUTRITION POSTS', 'Nutrition content to share with challenge participants'),
    ('2026 CHALLENGE TO DO LIST', 'Active to-do items for current/upcoming challenges'),
    ('2026 CHALLENGE SMS TO SEND', 'SMS queue for 2026 challenges'),
]
for name, desc in challenge_lists:
    p = doc.add_paragraph()
    run = p.add_run(f'{name}: ')
    run.bold = True
    p.add_run(desc)

doc.add_heading('Daily Challenge Operations (During Active Rounds)', level=2)
doc.add_paragraph(
    'During an active challenge round, the following daily calendar tasks apply:'
)
challenge_daily = [
    'Challenge Things (11:00am-12:00pm): Check the Challenge Trello board for tasks that need completing that day.',
    'Check RD Facebook Requests: Monitor the current round\'s Facebook group for new join requests daily.',
    'Be active on all challenge forums - interact with every client post (reception team can assist).',
]
for cd in challenge_daily:
    doc.add_paragraph(cd, style='List Bullet')

doc.add_heading('Challenge Timeline', level=2)
timeline = [
    '1 month before: Finalise challenge details, pricing, marketing materials',
    '3 weeks before: Begin promotion, send initial SMS, start sign-ups',
    '2 weeks before: Ramp up marketing, confirm all logistics',
    '1 week before: Send "week before challenge" template info to participants',
    'Weeks 1-7: Weekly tasks and SMS communications per Trello cards',
    'Post-challenge: Final SMS, results sharing, InBody scan comparisons, conversion to memberships',
]
for t in timeline:
    doc.add_paragraph(t, style='List Bullet')

doc.add_heading('Round 63 (RD63) - Completed Reference', level=2)
doc.add_paragraph('Final Day: 14 March 2026. Key post-challenge tasks completed:')
rd63_tasks = [
    ('5 Mar 2026', 'Post Nutrition Seminar'),
    ('14 Mar 2026', 'Final Day of Challenge - end of challenge photography (photo room, decorations, Roi photos), announce BOW (Best of Week) winner'),
    ('16 Mar 2026', 'Activate 21-day promo in MindBody; create challenge meal plans; edit results tiles in Canva'),
    ('17 Mar 2026', 'Announce takeover event details'),
    ('19 Mar 2026', 'Create meal plans for next round (multiple flavour bowls, same macros - see Sam\'s Neuform ambassador meal plans); Film Sam - "why our challenge works"'),
    ('20 Mar 2026', 'Remove all challengers from Cronometer; FROPRO delivery at 4pm'),
    ('23 Mar 2026', 'Deactivate 21-day promo in MindBody'),
    ('24 Mar 2026', 'HIIT Kids Promo'),
    ('26 Mar 2026', 'Send "week before challenge" template info for next round'),
]
rd63_table = doc.add_table(rows=1, cols=2)
rd63_table.style = 'Light Grid Accent 1'
hdr = rd63_table.rows[0].cells
hdr[0].text = 'Date'
hdr[1].text = 'Task'
for date, task in rd63_tasks:
    row = rd63_table.add_row().cells
    row[0].text = date
    row[1].text = task

doc.add_heading('Round 64 (RD64) - Upcoming Schedule', level=2)
doc.add_paragraph('This is the next active challenge round. All dates are confirmed in the Google Calendar.')
rd64_tasks = [
    ('18 Apr 2026', 'First Day of Challenge'),
    ('23 Apr 2026', 'How to Macro Seminar'),
    ('26 Apr 2026', 'Barbell Fundamentals Class'),
    ('15 May 2026', 'Midway Seminar with Chontel'),
    ('20 May 2026', 'Set up dates/calendar for next challenge round'),
    ('4 Jun 2026', 'Post Nutrition Seminar'),
    ('13 Jun 2026', 'Final Day of Challenge'),
]
rd64_table = doc.add_table(rows=1, cols=2)
rd64_table.style = 'Light Grid Accent 1'
hdr = rd64_table.rows[0].cells
hdr[0].text = 'Date'
hdr[1].text = 'Event / Task'
for date, task in rd64_tasks:
    row = rd64_table.add_row().cells
    row[0].text = date
    row[1].text = task

doc.add_heading('Key Challenge Processes', level=2)
doc.add_paragraph(
    'InBody scans at start and end of challenge. Ensure paper and wipes are stocked. '
    'Challenge tees to be organised. Meal plans via Cronometer (multiple flavour bowls, same macros). '
    'Results shared via Canva photo templates. Challenge dates and details confirmed in November for the following year.'
)
doc.add_paragraph(
    'Post-challenge admin: remove all challengers from Cronometer, deactivate any active MB promos, '
    'edit results tiles in Canva, set up next round calendar events and reminders, coordinate FROPRO delivery.'
)

doc.add_heading('Year-End Planning (November)', level=2)
doc.add_paragraph('Confirm for the following year:')
year_plan = [
    'Challenge dates',
    'HIIT Kids dates (discuss with Nelle)',
    'Promotions for the year',
    'Friday Takeovers continuing',
    'Challenge tees design and ordering',
    'Review what promos worked and what didn\'t',
    'Consider InBody scans to drive challenge participation',
]
for y in year_plan:
    doc.add_paragraph(y, style='List Bullet')

doc.add_page_break()

# ============================================================
# SECTION 14 - CONTENT & MARKETING
# ============================================================
doc.add_heading('14. Content & Marketing', level=1)

doc.add_heading('Content Ideas', level=2)
content = [
    'Retention content: reshare old transformation stories',
    'Advertising 10 class pass and casual passes',
    'Create facility walkthrough videos for pinned posts (downstairs gym and upstairs gym)',
    'Filming Sam and Chon: hero movements, fast/myths/lies, debunk nutrition facts',
    'Film HALF HALF change video with Sam and Chontel (calendar task)',
    'Film Sam - "why our challenge works" (scheduled post-challenge, see Section 13)',
]
for c in content:
    doc.add_paragraph(c, style='List Bullet')

doc.add_heading('Social Media Plan', level=2)
doc.add_paragraph(
    'Build portfolios on select members (transformation stories). '
    'Showcase coach journeys and community stories. '
    'Use testimonials for midway challenge mindset posts. '
    'Education content, testimonials, legacy content, and word-of-mouth strategy.'
)

doc.add_heading('Marketing Tactics', level=2)
marketing = [
    'Website update (contact Kassi for edits)',
    'MindBody widget integration on website',
    'Enquiry form automation and follow-up sequences',
    'Email automation via Active Campaign for promos',
    'Pre-screen form for first timers linked from website',
    'Walk-through video on website',
    'Testimonials on website and in email campaigns',
    'Paid ads strategy',
    'Tighten up enquiry-to-membership conversion process',
]
for m in marketing:
    doc.add_paragraph(m, style='List Bullet')

doc.add_page_break()

# ============================================================
# SECTION 15 - REVENUE GENERATION IDEAS
# ============================================================
doc.add_heading('15. Revenue Generation Ideas', level=1)

doc.add_heading('Schools - Year 12 Students', level=2)
doc.add_paragraph(
    'Create a pitch for Year 12 students on benefits of training with a coach. '
    '16/17 year olds don\'t need parental permission to train (only for contracts).'
)

doc.add_heading('Schools - Primary Classes', level=2)
doc.add_paragraph(
    'Explore what can be offered as additional revenue (Nelle - student classes). '
    'Model on Gumdale State School program. Coach Nelle to reach out to 3 schools per week. '
    'Talk to Kristen about how funding worked from the school side.'
)

doc.add_heading('Local Business Offers', level=2)
doc.add_paragraph('Develop a pitch for local businesses with corporate/group offers.')

doc.add_page_break()

# ============================================================
# SECTION 16 - FACILITY & EQUIPMENT
# ============================================================
doc.add_heading('16. Facility & Equipment Management', level=1)

doc.add_heading('Equipment List', level=2)
doc.add_paragraph(
    'The equipment list spreadsheet must be updated each time there are equipment faults, repairs, or replacements. '
    'This is vital to understand the lifetime of items and the need for ongoing maintenance.'
)
doc.add_paragraph('Spreadsheet: Google Drive (Equipment List)')

doc.add_heading('Printer', level=2)
doc.add_paragraph('Computer IP: 192.168.0.78 | Printer IP: 192.168.15.4 | Serial: #140375')
doc.add_paragraph('Contact for printer issues: Nicole Jaggs (contact in HIIT phone).')

doc.add_heading('WiFi', level=2)
doc.add_paragraph('HIIT_NOVA / HIIT_NOVA_5G - Password: <WIFI_PASSWORD>')

doc.add_heading('Non-Priority Facility Tasks', level=2)
facility_backlog = [
    'Edit reception manual (due April 2026)',
    'Paint kids area wall',
    'Rubber matting for stools upstairs',
    'Goals wall installation',
]
for f in facility_backlog:
    doc.add_paragraph(f, style='List Bullet')

doc.add_page_break()

# ============================================================
# SECTION 17 - GENERAL REFERENCE
# ============================================================
doc.add_heading('17. General Reference Information', level=1)

doc.add_heading('Business Details', level=2)
doc.add_paragraph('Company: HIIT Australia Pty Ltd')
doc.add_paragraph('ABN: 63 167 814 682')

doc.add_heading('Key Contacts', level=2)
contacts = [
    ('Michelle', 'Head Office / Finance - all purchases require her approval'),
    ('Janelle', 'Monthly gym walk-through partner'),
    ('Nelle', 'Coach - HIIT Kids, school programs'),
    ('Allexis', 'Reception team - envelope courier to Warehouse'),
    ('Sam', 'Owner/Director - DJ for events, content filming'),
    ('Chon', 'Operations - content filming, charity events'),
    ('Blair (PowerSupps)', 'blair@powersupps.com.au - supplements supplier'),
    ('Cameron (Eclipse Flooring)', 'cameron@eclipsefloorsolutions.com.au - flooring'),
    ('Physio Dynamics (Cleveland)', 'cleveland@physiodynamics.com.au / 0438264435 - WorkCover physio sessions'),
    ('Nicole Jaggs', 'Printer support (contact in HIIT phone)'),
    ('ARMA Group (Debt Collection)', 'consumer@armagroup.com.au / David@armagroup.com.au'),
]
table3 = doc.add_table(rows=1, cols=2)
table3.style = 'Light Grid Accent 1'
hdr = table3.rows[0].cells
hdr[0].text = 'Contact'
hdr[1].text = 'Role / Details'
for name, role in contacts:
    row = table3.add_row().cells
    row[0].text = name
    row[1].text = role

doc.add_heading('Physio Working at HIIT (WorkCover)', level=2)
doc.add_paragraph(
    'Physio from Physio Dynamics works with client Jarrad Towell, sessions once per week. '
    'Membership paid by the physio under WorkCover claim. Certificate of Currency on file. '
    'Physio is either Braydon or Tess. Bookings come through the system.'
)

doc.add_page_break()

# ============================================================
# SECTION 18 - NON-PRIORITY ACTIONS
# ============================================================
doc.add_heading('18. Non-Priority / Backlog Actions', level=1)
backlog = [
    ('Edit reception manual', 'Due April 2026'),
    ('Paint kids area wall', 'No due date'),
    ('Rubber matting for stools upstairs', 'No due date'),
    ('Goals wall', 'Reminder to not forget'),
    ('Filming Sam and Chon', 'Ideas: hero movements, myths/lies, debunk nutrition'),
]
for name, note in backlog:
    p = doc.add_paragraph()
    run = p.add_run(f'{name}: ')
    run.bold = True
    p.add_run(note)

doc.add_page_break()

# ============================================================
# SECTION 19 - COMPLETED ARCHIVE
# ============================================================
doc.add_heading('19. Completed Items Archive (2025-2026)', level=1)
doc.add_paragraph('The following items have been completed and are retained for reference:')

completed = [
    'Challenge Round 63 (RD63) completed - final day 14 Mar 2026, all post-challenge tasks actioned',
    'Social media plan creation (member portfolios, coach journeys)',
    'Christmas 2025 event (Saturday classes, BBQ, brekkie burgers, raffle, Children\'s Hospital donations)',
    'JotForm enquiry follow-up process established',
    'Vending machine follow-up',
    'Social events 2025 (family walks, community activities)',
    'Event calendar established for the year',
    'Charity events 2025 (Blood donation at Springwood - 20 spots)',
    'Black Friday sale (Dec free, 3-month $55/week, Nov 20 - Dec 30)',
    'Kids area plastering and painting completed (Oct 2025)',
    'Reception closure changes implemented',
    'Front door stickers updated via Signmart (ordered Aug 2025)',
    'Supplement sale organised (profit cap $10/product)',
    'Marketing tactics review and website update plan',
    'QR codes updated for individual cans',
    'Challenge photo template updated in Canva',
    'Remedial massage collaboration with Matt approved',
    'End of year challenge event planned',
    'Halloween kids disco and games event',
]
for c in completed:
    doc.add_paragraph(c, style='List Bullet')

# ============================================================
# SECTION 20 - AUTOMATION & INTEGRATION RECOMMENDATIONS
# ============================================================
doc.add_page_break()
doc.add_heading('20. Automation & Integration Recommendations', level=1)
doc.add_paragraph(
    'This section provides a detailed roadmap for automating HIIT Station operations. '
    'The vision: any team member can sit in the Office Manager role and run day-to-day operations '
    'by talking to Claude Code in natural language. Each recommendation below is mapped to a current '
    'manual process, with specific tools, setup steps, and difficulty ratings.'
)

# --- Helper function for recommendation blocks ---
def add_recommendation(title, task, how, tool, steps, difficulty):
    doc.add_heading(title, level=3)
    items = [
        ('What the task is', task),
        ('How it can be automated', how),
        ('Tool / Integration', tool),
        ('Difficulty', difficulty),
    ]
    for label, value in items:
        p = doc.add_paragraph()
        run = p.add_run(f'{label}: ')
        run.bold = True
        p.add_run(value)
    p = doc.add_paragraph()
    run = p.add_run('How to set it up:')
    run.bold = True
    for i, step in enumerate(steps, 1):
        doc.add_paragraph(f'{i}. {step}')

# ============================================================
# 20.1 - MINDBODY API AUTOMATIONS
# ============================================================
doc.add_heading('20.1 MindBody API Automations', level=2)
doc.add_paragraph(
    'MindBody offers a Public API (v6) and a Webhooks system. Once API access is activated (contact MindBody support, '
    'request API credentials for Site ID 5728999), Claude Code can be connected via a custom MCP server that wraps '
    'the MindBody REST API. This unlocks the highest-value automations in the business.'
)

add_recommendation(
    'Auto-Pull New Member Sign-Ups & Onboarding',
    'Every Monday the Office Manager manually runs Reports > Clients > New Members, adds names to Trello and a spreadsheet, then contacts each new member.',
    'Use the MindBody API GET /clients endpoint (filtered by CreationDate) to automatically pull new members daily. '
    'Auto-create a Trello card on the "2026 MEMBERSHIPS" list with member details, auto-add to the Google Sheet, '
    'and trigger a welcome SMS + email sequence.',
    'MindBody API + Claude MCP Server + Trello MCP + Google Sheets API',
    [
        'Request MindBody API credentials from MB support (Public API access for Site ID 5728999).',
        'Build a lightweight MCP server (Node.js or Python) that wraps the MB API - use the @anthropic-ai/sdk to define tools like get_new_members, get_client_details, get_attendance.',
        'Register the MCP server in Claude Code settings: ~/.claude/settings.json under mcpServers.',
        'Create a Claude Code slash command /new-members that calls GET /clients?CreationDateFrom={yesterday} and formats the results.',
        'Chain the output: auto-create Trello card via the existing Trello MCP, add row to Google Sheet via Google Sheets MCP.',
        'Set up the welcome email/SMS trigger (see Email and SMS sections below).',
        'Test with: "Claude, pull all new members from this week and add them to Trello and the sign-ups spreadsheet."',
    ],
    'Advanced (initial setup), then Easy (daily use via Claude)'
)

add_recommendation(
    'Attendance Tracking & Reporting',
    'Weekly and monthly class count reports are pulled manually from MB (Reports > Clients > Attendance Analysis). Sauna bookings checked separately in Calendly.',
    'Schedule a weekly API call to GET /classes and GET /visits to pull attendance data. Auto-generate a formatted report '
    'and post to the Office group chat or save as a Google Doc.',
    'MindBody API + Claude MCP Server + Google Docs API',
    [
        'Add get_attendance_report and get_class_visits tools to the MindBody MCP server.',
        'Define a /weekly-report slash command in Claude Code that pulls class counts by day-of-week, compares to previous week, and includes sauna bookings from Calendly API.',
        'Format output using the management report template from Section 5.',
        'Optionally auto-save to Google Drive as a dated document.',
        'Run with: "Claude, generate this week\'s attendance report."',
    ],
    'Medium'
)

add_recommendation(
    'Class Booking Confirmations & Reminders',
    'Currently managed through MindBody\'s built-in notifications, but these are limited in customisation.',
    'Use MB Webhooks (classRosterBooking.created) to trigger custom SMS/email confirmations with personalised messaging, '
    'coach info, and class-specific details.',
    'MindBody Webhooks + Twilio/MessageMedia (SMS) + ActiveCampaign (email)',
    [
        'Enable MindBody Webhooks in the MB Developer Portal.',
        'Set up a webhook listener (can be a simple cloud function on AWS Lambda or Google Cloud Functions).',
        'Subscribe to classRosterBooking.created and classRosterBooking.cancelled events.',
        'On booking: trigger a personalised SMS via MessageMedia API with class time, coach name, and location.',
        'On cancellation: send a rebooking prompt SMS.',
        'Connect webhook data to ActiveCampaign for email follow-ups.',
    ],
    'Advanced'
)

add_recommendation(
    'Payment Failure Alerts',
    'Office Manager manually checks MB Dashboard for failed autopays daily, then runs a Friday arrears report.',
    'Use MB Webhooks (sale.failed or scheduled daily API poll) to auto-detect failed payments. '
    'Instantly alert the Office Manager via Slack/email and auto-send a templated SMS to the member.',
    'MindBody API + MessageMedia + Slack/Email notification',
    [
        'Add a get_failed_payments tool to the MindBody MCP server using GET /sales with status filter.',
        'Create a /failed-payments slash command that pulls today\'s failures and shows a summary.',
        'Set up auto-SMS: on failure detection, send templated message from the account arrears message bank.',
        'Create a Trello card on the Office board for each failure requiring manual follow-up.',
        'For advanced setup: create a webhook listener for real-time failed payment events.',
        'Run with: "Claude, check for any failed payments today and text the members."',
    ],
    'Medium'
)

add_recommendation(
    'Challenge Participant Management',
    'Challengers are manually tracked in Cronometer, MB, Facebook groups, and Trello. Post-challenge cleanup (remove from Cronometer, deactivate promos, etc.) is manual.',
    'Build a challenge management workflow: API pulls all challenge participants from MB (by pricing option/service), '
    'tracks their progress, and automates post-challenge cleanup tasks.',
    'MindBody API + Cronometer API + Trello MCP + Claude Code',
    [
        'Tag all challenge participants in MB with a specific pricing option or membership type.',
        'Add a get_challenge_participants tool to the MCP server that filters by that pricing option.',
        'Create /challenge-status command that shows current participant count, attendance rates, and approaching end dates.',
        'Post-challenge: create /challenge-cleanup command that generates the task list (remove from Cronometer, deactivate promo, update Trello).',
        'Integrate with Cronometer API to auto-remove participants on challenge end date.',
        'Run with: "Claude, how many active challengers do we have and who hasn\'t attended this week?"',
    ],
    'Advanced'
)

add_recommendation(
    'BOW (Best of Week) Winner Data',
    'BOW winners are manually selected and announced during challenges.',
    'Auto-pull attendance and InBody data for the week, rank participants by attendance + body composition change, '
    'and present the top candidates for BOW selection.',
    'MindBody API + InBody/LookInBody API + Claude Code',
    [
        'Add tools to pull weekly attendance per challenge participant from MB.',
        'If LookInBody has an API, integrate scan result data (body fat %, muscle mass changes).',
        'Create /bow-candidates command that ranks participants by a weighted score (attendance + results).',
        'Present top 5 candidates to the Office Manager for final selection.',
        'Auto-generate the announcement post text and Canva template link.',
        'Run with: "Claude, who are the BOW candidates for this week?"',
    ],
    'Advanced'
)

doc.add_page_break()

# ============================================================
# 20.2 - EMAIL MARKETING
# ============================================================
doc.add_heading('20.2 Email Marketing Automation', level=2)
doc.add_paragraph(
    'Recommendation: Continue with ActiveCampaign (already in use). It has native MindBody integration, '
    'excellent automation workflows, and is well-suited for fitness businesses. '
    'Alternatives: Klaviyo (better for e-commerce/product sales), Mailchimp (simpler but less powerful automations). '
    'ActiveCampaign is the best fit for HIIT Station\'s needs given it\'s already set up.'
)

add_recommendation(
    'Automated Welcome Sequence for New Members',
    'New members are contacted manually via phone/SMS. No structured email onboarding exists.',
    'Create a 5-email welcome automation in ActiveCampaign triggered by MindBody new member webhook.',
    'ActiveCampaign + MindBody Integration',
    [
        'In ActiveCampaign, go to Automations > Create Automation.',
        'Set trigger: "Contact is added to list" (New Members list) or use the MindBody-ActiveCampaign native integration.',
        'Email 1 (Day 0): Welcome to HIIT Station - what to expect, facility guide, class schedule link.',
        'Email 2 (Day 2): Meet the coaches - introduce the team with photos.',
        'Email 3 (Day 5): First week check-in - how was your first class? Link to book next session.',
        'Email 4 (Day 10): Member perks - Friday Takeovers, challenges, sauna access, community forums.',
        'Email 5 (Day 14): Two-week milestone - encourage booking a regular schedule, introduce the challenge program.',
        'Connect MindBody to ActiveCampaign: Settings > Integrations > MindBody > enter Site ID and API credentials.',
        'Test the full sequence with a test contact.',
    ],
    'Easy (ActiveCampaign has drag-and-drop automation builder)'
)

add_recommendation(
    'Challenge Welcome & Weekly Check-In Emails',
    'Challenge communication is primarily via SMS and Facebook. No structured email touchpoints during the 8 weeks.',
    'Create a challenge-specific 8-week drip campaign with weekly nutrition tips, motivation, and key dates.',
    'ActiveCampaign',
    [
        'Create a new list: "Active Challengers" in ActiveCampaign.',
        'Build automation triggered on list addition (when challenge payment is processed in MB).',
        'Week 0: Welcome email with challenge guide, meal plan access, Cronometer setup instructions, Facebook group link.',
        'Weeks 1-7: Weekly email with that week\'s nutrition focus (pull from NUTRITION POSTS Trello list), workout tips, and the upcoming seminar/event.',
        'Week 4 (Midway): Special midway email with motivation, reminder of Midway Seminar with Chontel.',
        'Week 8: Final week email with photo day instructions, celebration details, and what\'s next.',
        'Post-challenge: Results email + re-enrolment offer for next round.',
        'Use ActiveCampaign\'s "Wait" steps between each weekly email.',
    ],
    'Easy'
)

add_recommendation(
    'Post-Challenge Results & Re-Enrolment Campaign',
    'Post-challenge follow-up is manual. No automated push to convert challengers into members or re-enrol for next round.',
    'Trigger a 3-email post-challenge sequence: results celebration, membership offer, next challenge early-bird.',
    'ActiveCampaign',
    [
        'Create automation triggered by challenge end date (use a date-based trigger or manual list move).',
        'Email 1 (Day 1 post-challenge): Congratulations + individual results summary (if InBody data available via API, personalise).',
        'Email 2 (Day 3): Special membership offer - "Keep your momentum" with a discounted conversion rate.',
        'Email 3 (Day 7): Next challenge early-bird registration with deadline.',
        'Add a conditional branch: if they purchase a membership or register for next challenge, stop the sequence.',
        'Track conversion rates in ActiveCampaign reporting.',
    ],
    'Easy'
)

doc.add_page_break()

# ============================================================
# 20.3 - SMS / TEXT COMMUNICATION
# ============================================================
doc.add_heading('20.3 SMS / Text Communication', level=2)
doc.add_paragraph(
    'Recommendation: MessageMedia (Australian-based, excellent API, local phone numbers, '
    'compliant with Australian spam regulations under the Spam Act 2003). '
    'Alternative: SMS Broadcast (already in use - consider migrating if API access is limited). '
    'Twilio is a strong technical option but US-based; MessageMedia offers better local support and compliance. '
    'SimpleTexting does not operate in Australia.'
)

add_recommendation(
    'Automated Class Reminders',
    'No automated class reminders beyond MindBody\'s built-in (limited) notifications.',
    'Send personalised SMS 2 hours before each booked class with coach name and any special instructions.',
    'MessageMedia API + MindBody Webhooks',
    [
        'Sign up for MessageMedia API access at messagemedia.com/au.',
        'Get API key and configure sender ID as "HIIT" (alphanumeric sender IDs supported in AU).',
        'Build a simple cloud function that: (a) polls MB for upcoming class bookings, (b) sends SMS 2 hours before class time.',
        'Alternatively, use MindBody webhook classRosterBooking.created to queue an SMS at class_time minus 2 hours.',
        'Template: "Hey {first_name}! Reminder: {class_name} at {time} today with {coach}. See you there! - HIIT Station"',
        'To connect via Claude Code: build a MessageMedia MCP server with send_sms and get_delivery_status tools.',
        'Run with: "Claude, send class reminders for tomorrow\'s sessions."',
    ],
    'Medium'
)

add_recommendation(
    'Challenge Day Reminders & Weekly SMS',
    'Challenge SMS are sent manually via SMS Broadcast following templates in the Trello board.',
    'Automate the entire challenge SMS calendar: pre-load all 8 weeks of messages and schedule them.',
    'MessageMedia API + Claude Code',
    [
        'Create a /schedule-challenge-sms command in Claude Code.',
        'When triggered, Claude reads the SMS templates from the Challenge Trello board (already connected).',
        'For each week, calculate the send date based on challenge start date.',
        'Use MessageMedia\'s scheduled sending API to queue all messages in advance.',
        'Include merge fields: {first_name}, {week_number}, {seminar_date}.',
        'Run with: "Claude, RD64 starts April 18. Schedule all challenge SMS for the 8 weeks."',
    ],
    'Medium'
)

add_recommendation(
    'BOW Announcements & Ad-Hoc SMS',
    'BOW winners and other announcements are manually composed and sent.',
    'Use Claude to draft and send announcements via MessageMedia in one command.',
    'MessageMedia API + Claude Code MCP',
    [
        'Build send_bulk_sms and send_individual_sms tools into the MessageMedia MCP server.',
        'Create contact lists synced from MindBody (active members, active challengers).',
        'Run with: "Claude, announce that Sarah Jones won BOW this week. Send to all RD64 challengers."',
        'Claude drafts the message, shows it for approval, then sends on confirmation.',
    ],
    'Easy (once MCP server is built)'
)

doc.add_page_break()

# ============================================================
# 20.4 - CALENDAR AUTOMATION
# ============================================================
doc.add_heading('20.4 Calendar Automation', level=2)

add_recommendation(
    'Auto-Create Challenge Round Events from Template',
    'Challenge round events (seminars, final day, admin tasks) are manually created in Google Calendar each round.',
    'Define a challenge template with relative dates (e.g., "Day 1", "Day 5 = How to Macro Seminar", "Day 28 = Midway Seminar"). '
    'When a new round is set, auto-populate all events.',
    'Google Calendar MCP (already connected) + Claude Code',
    [
        'Create a challenge event template as a JSON file in the project folder with entries like: {"day_offset": 0, "title": "RD{round} - First Day of Challenge", "time": "05:00", "duration_hours": 14}.',
        'Create a /setup-challenge-calendar slash command that takes round number and start date as arguments.',
        'Claude reads the template, calculates actual dates, and creates all events via the Google Calendar MCP.',
        'Include recurring daily events: "Challenge Things 11am-12pm" and "Check RD{round} Facebook requests".',
        'Run with: "Claude, set up the calendar for RD65 starting August 3."',
    ],
    'Easy'
)

add_recommendation(
    'Automated Admin Task Reminders',
    'Recurring admin reminders (debt collection review, incident report, gym walk-through, sleeper check) are in the calendar but were manually set up.',
    'Use Claude to audit and maintain the recurring calendar events, ensuring nothing is missed when dates shift.',
    'Google Calendar MCP + Claude Code',
    [
        'Create a /check-recurring-tasks command that lists all recurring admin events for the next 30 days.',
        'Claude cross-references against the operations manual task list (Sections 5 and 6) to flag any missing reminders.',
        'Auto-create any missing events.',
        'Run with: "Claude, are all my monthly recurring tasks set up in the calendar for next month?"',
    ],
    'Easy'
)

add_recommendation(
    'Sync MindBody Schedule with Google Calendar',
    'MindBody class schedule and Google Calendar are separate systems. Changes in one don\'t reflect in the other.',
    'Daily sync of MB class schedule to a dedicated Google Calendar so the team has one view of everything.',
    'MindBody API + Google Calendar MCP + Claude Code (or Zapier as a no-code alternative)',
    [
        'Add a get_class_schedule tool to the MindBody MCP server.',
        'Create a /sync-schedule command that pulls tomorrow\'s MB classes and creates/updates Google Calendar events.',
        'Use a dedicated calendar (e.g., "HIIT Classes") to avoid cluttering the main calendar.',
        'Run daily as part of the morning routine.',
        'No-code alternative: Use Zapier\'s MindBody trigger + Google Calendar action for automatic sync.',
    ],
    'Medium'
)

doc.add_page_break()

# ============================================================
# 20.5 - TRELLO AUTOMATION
# ============================================================
doc.add_heading('20.5 Trello Automation', level=2)

add_recommendation(
    'Auto-Create Challenge Checklist Cards',
    'At the start of each challenge round, cards are manually created on the Challenge Trello board with weekly checklists.',
    'Use Claude + Trello MCP to auto-generate all 8 weeks of cards from a template at the start of each round.',
    'Trello MCP (already connected) + Claude Code',
    [
        'Define a challenge card template (JSON or in a Trello template list) with card names, descriptions, and checklist items for each week.',
        'Create a /setup-challenge-trello slash command.',
        'Claude reads the template and creates cards on the Challenge board: "Week 1 - TO DO", "Week 2 - TO DO", etc., each with the appropriate checklist.',
        'Also create the SMS communication cards for each week.',
        'Run with: "Claude, set up the Trello board for RD65."',
    ],
    'Easy'
)

add_recommendation(
    'Task Completion Tracking & Reporting',
    'Task completion is tracked by checking Trello manually. No summary reporting.',
    'Use Claude to generate a daily/weekly task completion report from Trello card status.',
    'Trello MCP + Claude Code',
    [
        'Create a /task-report command that scans all active lists on the HIIT Office board.',
        'For each card: check due dates, completion status, and flag overdue items.',
        'Generate a summary: X tasks completed, Y overdue, Z due this week.',
        'Run with: "Claude, give me a task status update for this week."',
    ],
    'Easy'
)

add_recommendation(
    'Trello Butler Automation Rules',
    'Moving cards between lists, setting due dates, and adding labels is manual.',
    'Set up Trello Butler (built-in, free) rules to automate common card movements and notifications.',
    'Trello Butler (built into Trello, no additional cost)',
    [
        'Go to Trello > Automation (Butler icon in top menu).',
        'Rule 1: When a card is moved to "COMPLETED - 2026", automatically mark the due date as complete and add a green label.',
        'Rule 2: When a card\'s due date is within 2 days, move it to the top of its list and add a red "urgent" label.',
        'Rule 3: On the 1st of each month, auto-create the monthly membership tracking cards (cancellations + sign-ups) from a template.',
        'Rule 4: When a checklist item is completed on a challenge card, post a comment with the completion timestamp.',
        'Calendar command: Create a recurring monthly command that creates the monthly report card on the first Monday.',
    ],
    'Easy'
)

doc.add_page_break()

# ============================================================
# 20.6 - SOCIAL MEDIA & CONTENT
# ============================================================
doc.add_heading('20.6 Social Media & Content Automation', level=2)

add_recommendation(
    'Social Media Scheduling',
    'Posts are created in Canva and manually posted to Instagram and Facebook. Timing is inconsistent.',
    'Use a scheduling tool to batch-create and auto-publish content on a weekly basis.',
    'Later.com (recommended for Instagram-first businesses; free plan available; visual calendar; Canva integration)',
    [
        'Sign up at later.com and connect the HIIT Capalaba Instagram and Facebook accounts.',
        'In Canva, use the "Schedule" feature to export designs directly to Later (Canva Pro has native Later integration).',
        'Set up a weekly content calendar: Monday = motivation/transformation, Wednesday = tips/education, Friday = event promo/takeover.',
        'Use Later\'s "Best Time to Post" feature for Australian timezone optimisation.',
        'Batch-create 2 weeks of content at a time during a dedicated content session.',
        'Alternative: Buffer (simpler interface, also has a free plan) or Meta Business Suite (free, built-in, but less visual).',
    ],
    'Easy'
)

add_recommendation(
    'Auto-Posting Challenge Results Tiles',
    'Results tiles are manually created in Canva and posted after each challenge.',
    'Create a semi-automated pipeline: Canva template + data merge + scheduled posting.',
    'Canva + Later + Claude Code',
    [
        'Set up a Canva template for results tiles with placeholder text fields (name, before/after stats, photo).',
        'Use Canva\'s Bulk Create feature: upload a CSV with participant names and results data to auto-generate all tiles.',
        'Export all tiles to Later and schedule posts across the week following challenge end.',
        'Claude can help: "Claude, generate the results CSV for RD64 from the InBody data and challenge participant list."',
    ],
    'Medium'
)

add_recommendation(
    'Content Calendar Automation',
    'No structured content calendar exists. Content ideas are scattered across Trello cards.',
    'Consolidate all content planning into a Trello board with automated recurring cards and calendar sync.',
    'Trello MCP + Google Calendar MCP + Claude Code',
    [
        'Create a "HIIT Content Calendar" Trello board with lists for each week.',
        'Use Butler to auto-create weekly content cards with a checklist: Monday post, Wednesday post, Friday post, Stories x3.',
        'Create a /content-plan command: "Claude, create next week\'s content plan" - Claude generates card descriptions with post ideas based on upcoming events (Takeover themes, challenge milestones, public holidays).',
        'Sync content due dates to Google Calendar for visibility.',
    ],
    'Easy'
)

doc.add_page_break()

# ============================================================
# 20.7 - REPORTING & ADMIN
# ============================================================
doc.add_heading('20.7 Reporting & Admin Automation', level=2)

add_recommendation(
    'Weekly Automated Reports from MindBody',
    'The Friday management report is manually compiled from multiple MB reports, Calendly, and spreadsheets.',
    'Auto-generate the full weekly report in one command, pulling from all data sources.',
    'MindBody MCP + Calendly API + Google Sheets API + Claude Code',
    [
        'Build the /weekly-report slash command to orchestrate: (a) MB attendance data, (b) MB new members, (c) MB failed payments, (d) MB cancellations, (e) Calendly sauna bookings.',
        'Claude formats the data into the management report template (Section 5).',
        'Auto-calculate: total members = total - staff - challenge-only + declined + suspended.',
        'Output as formatted text for the group chat AND save to Google Drive as a dated document.',
        'Run with: "Claude, generate the weekly management report."',
    ],
    'Medium'
)

add_recommendation(
    'Cronometer Member Management',
    'Challengers are manually added to and removed from Cronometer at the start/end of each round.',
    'Automate the add/remove process using Cronometer\'s Gold subscription API (if available) or a structured CSV workflow.',
    'Cronometer API (if available) or CSV batch process + Claude Code',
    [
        'Check if Cronometer offers API access for professional/business accounts.',
        'If API available: build a Cronometer MCP server with add_client and remove_client tools.',
        'If no API: Create a /challenge-cronometer command that generates a CSV of participants to add (from MB data) and a list to remove (from previous round).',
        'Use the CSV to batch-process in Cronometer\'s web interface.',
        'Run with: "Claude, generate the Cronometer add list for RD65 and the remove list for RD64."',
    ],
    'Medium (depends on API availability)'
)

add_recommendation(
    'Meal Plan Distribution',
    'Meal plans are created by Sam (using Neuform ambassador meal plans as reference, multiple flavour bowls, same macros) and manually distributed.',
    'Automate distribution via email and/or a shared Google Drive folder with automatic access grants.',
    'ActiveCampaign + Google Drive API + Claude Code',
    [
        'Create meal plan PDFs and store in a structured Google Drive folder: /Challenges/RD{round}/Meal Plans/.',
        'When a challenger signs up (detected via MB API), auto-share the Google Drive folder with their email.',
        'Simultaneously trigger the ActiveCampaign challenge welcome email which includes the meal plan access link.',
        'Create a /distribute-meal-plans command: Claude gets the participant list from MB, shares the Drive folder, and triggers the email.',
        'Run with: "Claude, distribute the RD65 meal plans to all registered challengers."',
    ],
    'Medium'
)

doc.add_page_break()

# ============================================================
# 20.8 - CLAUDE AS THE OPERATOR
# ============================================================
doc.add_heading('20.8 Claude as the Operator', level=2)
doc.add_paragraph(
    'The ultimate goal is that a non-technical person can manage HIIT Station\'s entire operation by talking to Claude Code. '
    'Below is the architecture, the MCP servers needed, and a daily workflow script.'
)

doc.add_heading('MCP Server Architecture', level=3)
doc.add_paragraph('Claude Code connects to external systems via MCP (Model Context Protocol) servers. Here is the target setup:')

mcp_servers = [
    ('Trello MCP', 'Connected', 'Task management, challenge boards, membership tracking'),
    ('Google Calendar MCP', 'Connected', 'Scheduling, meetings, challenge events, reminders'),
    ('MindBody MCP', 'To Build', 'Member management, attendance, payments, reporting, class schedules'),
    ('Gmail MCP', 'To Connect', 'Email monitoring, sending follow-ups, contract reminders'),
    ('Google Sheets MCP', 'To Connect', 'Spreadsheet read/write for cancellations, debt collection, equipment tracking'),
    ('Google Drive MCP', 'To Connect', 'Document management, meal plan distribution, report storage'),
    ('MessageMedia MCP', 'To Build', 'SMS sending, bulk messaging, delivery tracking'),
    ('Canva MCP', 'Available', 'Design generation, results tiles, social media posts'),
    ('Slack MCP', 'Optional', 'Team notifications (alternative to group chat SMS wrap-ups)'),
]
mcp_table = doc.add_table(rows=1, cols=3)
mcp_table.style = 'Light Grid Accent 1'
hdr = mcp_table.rows[0].cells
hdr[0].text = 'MCP Server'
hdr[1].text = 'Status'
hdr[2].text = 'Purpose'
for name, status, purpose in mcp_servers:
    row = mcp_table.add_row().cells
    row[0].text = name
    row[1].text = status
    row[2].text = purpose

doc.add_heading('Setting Up for a Non-Technical Operator', level=3)
setup_steps = [
    'Install Claude Code on the office Mac (one-time setup by Sam or a developer).',
    'Configure all MCP servers in ~/.claude/settings.json with API credentials.',
    'Create a CLAUDE.md file in the project directory with HIIT-specific context: business rules, common commands, and this operations manual as reference.',
    'Create custom slash commands for common operations: /morning, /new-members, /failed-payments, /weekly-report, /challenge-status, /task-report.',
    'Train the operator: show them they can type natural language like "What needs to be done today?" and Claude will check Trello, Calendar, and MindBody to give a prioritised task list.',
    'Set up Claude Code hooks (in settings.json) for automated daily checks that run when Claude Code starts.',
]
for i, s in enumerate(setup_steps, 1):
    doc.add_paragraph(f'{i}. {s}')

doc.add_heading('Suggested Daily Workflow: What to Say to Claude Each Morning', level=3)
doc.add_paragraph(
    'Below is a suggested script for the Office Manager to follow each morning. '
    'Simply type these commands (or similar natural language) into Claude Code:'
)

morning_workflow = [
    ('"Good morning Claude, run my daily briefing."',
     'Claude checks: today\'s Google Calendar events, overdue Trello cards, any MindBody alerts (failed payments, new members), '
     'and the Challenge Trello board if a round is active. Presents a prioritised task list.'),
    ('"Pull yesterday\'s new leads and first-time visitors."',
     'Claude runs the MB First Visit and Lead Management reports, formats the results, and adds to the Enquiries spreadsheet.'),
    ('"Any failed payments to chase today?"',
     'Claude checks MB for failed autopays, lists affected members, and drafts SMS messages for approval.'),
    ('"Check the challenge board - what\'s due today?"',
     'Claude reads the active challenge round\'s Trello cards and lists uncompleted checklist items for today.'),
    ('"Send the weekly wrap-up to the staff chat."',
     '(Friday only) Claude compiles the week\'s data into the wrap-up format and presents it for review before sending.'),
    ('"Generate this month\'s management report."',
     '(Monthly) Claude pulls all data sources and generates the full monthly report, ready for the Management Team Meeting.'),
    ('"Set up the calendar and Trello for the next challenge round, RD65, starting August 3."',
     'Claude creates all calendar events and Trello cards from templates, schedules all SMS, and sets up the email automation trigger.'),
]

for cmd, result in morning_workflow:
    p = doc.add_paragraph()
    run = p.add_run(cmd)
    run.bold = True
    run.italic = True
    doc.add_paragraph(f'  {result}')

doc.add_heading('Implementation Priority Roadmap', level=3)
doc.add_paragraph('Recommended order of implementation based on impact and ease:')

roadmap = [
    ('Phase 1 - Quick Wins (Week 1-2)', [
        'Set up Trello Butler rules for auto-labelling and card creation',
        'Create ActiveCampaign welcome email sequence for new members',
        'Create ActiveCampaign challenge email drip campaign',
        'Set up Later.com for social media scheduling',
        'Create challenge calendar template for Claude Code',
    ]),
    ('Phase 2 - Core Integrations (Week 3-6)', [
        'Build MindBody MCP server (new members, attendance, failed payments)',
        'Connect Gmail MCP for email monitoring',
        'Connect Google Sheets MCP for spreadsheet automation',
        'Set up MessageMedia account and build MCP server for SMS',
        'Create core slash commands: /morning, /new-members, /failed-payments',
    ]),
    ('Phase 3 - Advanced Automation (Week 7-12)', [
        'Build full weekly report automation',
        'Set up MindBody webhooks for real-time alerts',
        'Automate challenge setup end-to-end (calendar + Trello + SMS + email)',
        'Build Cronometer integration for challenge participant management',
        'Create the full daily briefing command',
    ]),
    ('Phase 4 - Optimisation (Ongoing)', [
        'Refine automations based on team feedback',
        'Add InBody API integration for challenge results',
        'Build post-challenge conversion automation',
        'Automate meal plan distribution',
        'Train all staff on Claude Code daily workflow',
    ]),
]

for phase_name, items in roadmap:
    doc.add_heading(phase_name, level=4)
    for item in items:
        doc.add_paragraph(item, style='List Bullet')

doc.add_page_break()

# ============================================================
# SUMMARY TABLE
# ============================================================
doc.add_heading('20.9 Full Recommendations Summary', level=2)
summary_data = [
    ('New Member Onboarding', 'MindBody API + Trello MCP', 'Advanced'),
    ('Attendance Reporting', 'MindBody API + Google Docs', 'Medium'),
    ('Class Booking Reminders', 'MB Webhooks + MessageMedia', 'Advanced'),
    ('Payment Failure Alerts', 'MindBody API + MessageMedia', 'Medium'),
    ('Challenge Participant Mgmt', 'MindBody API + Cronometer', 'Advanced'),
    ('BOW Winner Data', 'MindBody API + InBody', 'Advanced'),
    ('Welcome Email Sequence', 'ActiveCampaign', 'Easy'),
    ('Challenge Email Drip', 'ActiveCampaign', 'Easy'),
    ('Post-Challenge Re-Enrolment', 'ActiveCampaign', 'Easy'),
    ('Automated Class Reminders (SMS)', 'MessageMedia API', 'Medium'),
    ('Challenge SMS Scheduling', 'MessageMedia + Claude', 'Medium'),
    ('BOW Announcement SMS', 'MessageMedia + Claude', 'Easy'),
    ('Challenge Calendar Template', 'Google Calendar MCP + Claude', 'Easy'),
    ('Admin Task Reminders Audit', 'Google Calendar MCP + Claude', 'Easy'),
    ('MB-to-Calendar Sync', 'MindBody API + Calendar MCP', 'Medium'),
    ('Challenge Trello Auto-Setup', 'Trello MCP + Claude', 'Easy'),
    ('Task Completion Reporting', 'Trello MCP + Claude', 'Easy'),
    ('Trello Butler Rules', 'Trello Butler (built-in)', 'Easy'),
    ('Social Media Scheduling', 'Later.com', 'Easy'),
    ('Results Tiles Pipeline', 'Canva + Later + Claude', 'Medium'),
    ('Content Calendar', 'Trello + Calendar + Claude', 'Easy'),
    ('Weekly MB Report', 'MindBody MCP + Claude', 'Medium'),
    ('Cronometer Automation', 'Cronometer API + Claude', 'Medium'),
    ('Meal Plan Distribution', 'ActiveCampaign + Google Drive', 'Medium'),
    ('Claude Daily Briefing', 'All MCP Servers + Claude', 'Medium'),
]

summary_table = doc.add_table(rows=1, cols=3)
summary_table.style = 'Light Grid Accent 1'
hdr = summary_table.rows[0].cells
hdr[0].text = 'Automation'
hdr[1].text = 'Tools'
hdr[2].text = 'Difficulty'
for auto, tools, diff in summary_data:
    row = summary_table.add_row().cells
    row[0].text = auto
    row[1].text = tools
    row[2].text = diff

doc.add_page_break()

# --- Final page ---
doc.add_paragraph()
doc.add_paragraph()
final = doc.add_paragraph()
final.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = final.add_run('End of Operations Manual')
run.bold = True
run.font.size = Pt(16)

note = doc.add_paragraph()
note.alignment = WD_ALIGN_PARAGRAPH.CENTER
note.add_run(
    '\nThis document was generated from the HIIT Office and HIIT Challenge Trello boards\n'
    'and the hiitstationmealplan@gmail.com Google Calendar.\n'
    'For the most current information, always refer to the live Trello boards, Google Calendar, and MindBody system.'
)

# Save
doc.save('/Users/samduncan/Desktop/claude-mcp/HIIT-Operations-Manual.docx')
print('Document saved successfully.')
