"""
Lead automation: reads MindBody new lead emails from Erin's Outlook inbox,
extracts the lead's email and name, and creates a draft reply using the
membership enquiries template.

Runs daily at 5am AEST via a scheduled endpoint.
"""

import re
import logging

logger = logging.getLogger(__name__)

ERIN_EMAIL = "admin@hiitaustralia.com.au"
MINDBODY_SENDER = "noreply@mail.mindbodyemail.com"

TEMPLATE_SUBJECT = "You’re officially on our radar \U0001f440 Let’s get you into HIIT Capalaba"

TEMPLATE_BODY_HTML = """\
<p>Hey [Name],</p>
<p>Welcome to HIIT!! We’re reaching out to help you turn that first step into your first session.</p>
<p>Whether you’re looking to jump into a membership or go all-in with our 8-Week Challenge, we’ve got options to suit where you’re at right now. This isn’t just a gym. It’s a community that shows up, backs each other, and gets results together.</p>
<p><b>\U0001f525 Memberships</b></p>
<p>If you want to try us out first, we have our casual pass for $20 per session. We also have a 10 class pass for $180, which gets you one session free.</p>
<p>If you’re keen to get moving now, this is the easiest way to experience what HIIT is all about.</p>
<p><b>How to book:</b></p>
<ol>
<li>Download the HIIT Station App</li>
<li>Create your account</li>
<li>Tap BUY, Intro &amp; Offers, Casual Pass</li>
<li>Book into a class and you’re good to go</li>
</ol>
<p>Ready to commit? Memberships can also be purchased directly in the app. A breakdown of our current membership options is below with additional info in the link.</p>
<p>\U0001f449 Membership Handbook: <a href="https://hiitgames.my.canva.site/membership-capalaba">Click here for more membership info</a></p>
<p><b>\U0001f4a5 8-Week Challenge</b></p>
<p>If you love structure, accountability, and being part of something big, our 8-Week Challenge is where the magic happens. Our Challenges run 4 times per year and book out fast. They’re fully supported, structured, and designed to get real results.</p>
<p>\U0001f449 Challenge Handbook: <a href="https://hiitgames.my.canva.site/challenge-handbook-capalaba-website">Click here for 8 week challenge info</a></p>
<p><b>How to secure your Challenge spot:</b></p>
<ol>
<li>Complete the prescreen form here: \U0001f449 <a href="https://hiit-registration.vercel.app/">Challenge Pre-Screen</a></li>
<li>Pay your deposit at the end to lock in your place</li>
<li>Download the HIIT Station App</li>
<li>Create your profile so we can set you up for success</li>
</ol>
<p>Once that’s done, we’ll guide you through everything else, you’re never left guessing.</p>
<p><b>\U0001f90d Not Sure Which Path to Take?</b></p>
<p>That’s totally okay, most people start exactly where you are now.</p>
<p>If you want to chat it through, ask questions, or get help choosing what’s right for you, give us a call on 0421 188 443. We’re always happy to help.</p>
<p>We’d absolutely love to see you in the gym, meet you in person, and help you build momentum toward feeling stronger, fitter, and more confident.</p>
<p>Speak soon,</p>
<p>The HIIT Capalaba Team \U0001f4aa</p>
"""


def _extract_lead_email(email_body):
    """Extract the lead's email address from a MindBody notification email body."""
    skip = {"noreply@mail.mindbodyemail.com", "noreply@mindbodyonline.com"}
    emails_found = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', email_body)
    for addr in emails_found:
        addr_lower = addr.lower()
        if addr_lower not in skip and "mindbody" not in addr_lower and "hiitaustralia" not in addr_lower:
            return addr
    return None


def _extract_lead_name(email_body):
    """Try to extract the lead's first name from a MindBody notification email body."""
    patterns = [
        r'Client\s*name\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'(?:First\s*Name|Name)\s*[:\-]\s*([A-Z][a-z]+)',
        r'(?:Lead|Member)\s*[:\-]\s*([A-Z][a-z]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, email_body, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            return name.split()[0] if name else name

    for line in email_body.split('\n'):
        line = line.strip()
        name_match = re.match(r'^([A-Z][a-z]+)\s+[A-Z][a-z]+$', line)
        if name_match:
            return name_match.group(1)

    return None


_processed_ids = set()


def process_new_leads(include_read=False):
    """Check Erin's inbox for MindBody lead emails and create draft replies.

    Args:
        include_read: If True, process already-read emails too (for testing).

    Returns a list of dicts with details of each draft created.
    """
    from outlook_helper import read_inbox, read_email, create_draft

    emails = read_inbox(count=20, search=MINDBODY_SENDER, outlook_user=ERIN_EMAIL)
    logger.info(f"Lead automation: found {len(emails)} emails matching MindBody search")

    mindbody_emails = [
        e for e in emails
        if MINDBODY_SENDER in e.get("from_email", "").lower()
        and "lead" in e.get("subject", "").lower()
        and (include_read or not e["is_read"])
        and e["id"] not in _processed_ids
    ]

    if not mindbody_emails:
        logger.info("No new MindBody lead emails found")
        return []

    drafts_created = []

    for email_summary in mindbody_emails:
        try:
            full_email = read_email(email_summary["id"], outlook_user=ERIN_EMAIL)
            body = full_email.get("body", "")

            lead_email = _extract_lead_email(body)
            if not lead_email:
                logger.warning(f"Could not extract lead email from: {email_summary['subject']}")
                continue

            lead_name = _extract_lead_name(body)
            personalised_body = TEMPLATE_BODY_HTML.replace("[Name]", lead_name or "there")

            create_draft(
                to=lead_email,
                subject=TEMPLATE_SUBJECT,
                body=personalised_body,
                outlook_user=ERIN_EMAIL,
                body_type="html",
            )

            drafts_created.append({
                "lead_email": lead_email,
                "lead_name": lead_name or "Unknown",
                "draft_subject": TEMPLATE_SUBJECT,
                "original_subject": email_summary["subject"],
            })

            _processed_ids.add(email_summary["id"])
            logger.info(f"Draft created for lead: {lead_email} ({lead_name})")

        except Exception as e:
            logger.error(f"Failed to process lead email '{email_summary['subject']}': {e}")

    return drafts_created


def format_lead_summary(drafts):
    """Format the lead automation results for WhatsApp notification."""
    if not drafts:
        return "*Lead Automation*\nNo new MindBody leads found this morning."

    lines = [f"*Lead Automation*\n{len(drafts)} new lead draft{'s' if len(drafts) != 1 else ''} created\n"]
    for i, d in enumerate(drafts, 1):
        lines.append(
            f"{i}. *{d['lead_name']}*\n"
            f"   Email: {d['lead_email']}\n"
            f"   From: {d['original_subject']}"
        )

    lines.append(f"\nDrafts are ready in Erin's Outlook.")
    return "\n".join(lines)
