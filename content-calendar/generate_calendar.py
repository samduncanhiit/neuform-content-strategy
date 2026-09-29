#!/usr/bin/env python3
"""Generate the remaining 27 days of the Neuform content calendar (Days 4-30)."""
import json

with open('/tmp/examples_by_cat.json') as f:
    examples = json.load(f)

# Track which examples we've used to avoid repeats
used_urls = set()

def get_examples(category, count=3):
    """Get unused examples from a category."""
    results = []
    for v in examples.get(category, []):
        if v['url'] not in used_urls and len(results) < count:
            used_urls.add(v['url'])
            results.append(v)
    # If not enough, pull from related categories
    if len(results) < count:
        fallbacks = {'workout': 'transformation', 'mumlife': 'motivation', 'nutrition': 'workout',
                     'transformation': 'mumlife', 'motivation': 'workout', 'couple': 'workout'}
        fb = fallbacks.get(category, 'workout')
        for v in examples.get(fb, []):
            if v['url'] not in used_urls and len(results) < count:
                used_urls.add(v['url'])
                results.append(v)
    return results

def example_html(exs, learnings):
    """Generate example HTML with strategic annotations."""
    html = '      <div class="cal-examples">\n'
    html += '        <div class="ex-label">Study These Examples (verified 500K+ views)</div>\n'
    for i, ex in enumerate(exs):
        learn = learnings[i] if i < len(learnings) else "Study the hook, pacing, and CTA structure."
        html += f'        <a href="{ex["url"]}" target="_blank">@{ex["author"]} &mdash; {ex["views"]} views &rarr;</a>\n'
        html += f'        <div class="cal-option">{ex["cap"][:60]}... <strong>Learn:</strong> {learn}</div>\n'
    html += '      </div>\n'
    return html

def script_html(hooks, script_body, label="Voiceover Script"):
    """Generate script block with 3 hook options."""
    html = '      <div class="cal-script">\n'
    html += f'        <span class="script-label">{label} &mdash; Choose 1 of 3 Hooks</span>\n\n'
    for i, (htype, hook) in enumerate(hooks):
        html += f'        <div class="hook-option"><strong>Hook {chr(65+i)} ({htype}):</strong> "{hook}"</div>\n'
    html += f'\n        <div class="script-body">\n{script_body}\n        </div>\n'
    html += '      </div>\n'
    return html

# === CONTENT PLAN FOR DAYS 4-30 ===
# Each day: [date, day_name, posts]
# Each post: {platform, fmt, pillar, title, caption, options, execution, examples_cat, example_learnings, script (optional), cta}

days = [
    # === WEEK 1 CONTINUED ===
    {"date": "April 4", "day": "Saturday", "week": None, "posts": [
        {
            "platform": "ig", "fmt": "Reel", "pillar": "mumlife",
            "title": "Saturday morning as a mum of 5 who still trains",
            "caption": """Saturday morning.\n\nMost people sleep in.\nI've already:\n&bull; Made 5 breakfasts\n&bull; Refereed 3 fights\n&bull; Found a missing shoe (it was in the fridge)\n&bull; Loaded the car\n&bull; Hit the gym\n\nIt's 8:47am.\n\nThis is the life. Chaotic, loud, exhausting &mdash; and I wouldn't change a thing.\n\nTag a mum who gets it.\n\n#mumlife #fitmum #musclemummy #saturdaymorning #neuform""",
            "script": {
                "label": "Voiceover Script",
                "hooks": [
                    ("Humour", "It's Saturday morning. I've been awake since five thirty. I've already done more than most people do all day."),
                    ("Relatable", "You know what Saturday looks like when you've got five kids? Chaos. Beautiful, exhausting chaos."),
                    ("Direct", "People romanticise the fit mum life. Let me show you what it actually looks like.")
                ],
                "body": """<em>[footage: morning chaos montage — kids, breakfast mess, finding shoe]</em>\n\nThis is Saturday in our house. It's loud. It's messy. Someone's always crying — sometimes it's me.\n\n<em>[footage: grabbing gym bag, quick gym session]</em>\n\nBut I still make it happen. Not because I'm superwoman. Because this is how I stay sane.\n\n<em>[footage: post-gym, coffee, kids playing]</em>\n\nThe gym isn't separate from motherhood. It's what makes me better AT motherhood.\n\nTag a mum who gets it."""
            },
            "options": [
                ("Option B", '"The unglamorous reality of being a fit mum" — expectation vs reality format'),
                ("Option C", '"Saturday morning GRWM — mum of 5 edition" — get ready with me but chaotic and real')
            ],
            "execution": """<strong>Filming:</strong> Film in REAL TIME as your Saturday unfolds. Phone in hand, capture the chaos. The messier and more real, the better — brand guidelines say "use the imperfect shots."<br><br><strong>Editing:</strong> 30-45 seconds. Fast cuts synced to upbeat music. Text overlays timestamping the morning (5:30am, 6:15am, 7:00am etc). End with a calm gym moment — the contrast IS the content.<br><br><strong>Posting:</strong> 9am AEST Saturday. Mums are scrolling while kids play. This is peak share-to-friend time.<br><br><strong>Why this works:</strong> Weekend lifestyle content gets the highest share rate. Mums tag other mums. Every tag = free distribution to someone in your exact target audience.""",
            "examples_cat": "mumlife",
            "example_learnings": [
                "Real mum content with morning routine chaos. The RELATABLE factor drives shares — mums tag mums who 'get it.' This is free distribution to your exact target audience.",
                "Postpartum fitness journey documented honestly. The journey IS the content. Chontel's 5-kid version is even more powerful.",
                "SAHM motivation that shows the real struggle + the reward. Not performative — genuine. This builds the trust that converts to app downloads."
            ],
            "cta": "Tag a mum who gets it"
        },
        {
            "platform": "tk", "fmt": "Video", "pillar": "motivation",
            "title": "POV: You finally start the program you've been putting off",
            "caption": """POV: You finally stop saying "I'll start Monday" and you actually start.\n\nMonday comes.\nYou show up.\nIt's hard. But you show up again.\nAnd again.\n\nAnd suddenly you're not the person who "wants to get fit."\nYou're the person who IS fit.\n\nYour future self is begging you to start today.\n\nComment START and I'll send you my program.\n\n#fyp #motivation #gym #fitness #neuform #fitmum""",
            "options": [
                ("Option B", '"The difference between wanting it and doing it" — split screen of scrolling on couch vs training'),
                ("Option C", '"A letter to the version of you that hasn\'t started yet" — emotional, direct-to-camera')
            ],
            "execution": """<strong>Filming:</strong> Aspirational montage. Use your strongest training footage — the lifts that look impressive, the physique shots, the determination. This is about making viewers SEE themselves doing it.<br><br><strong>Editing:</strong> Under 15 seconds. Trending motivational sound. Quick cuts on the beat. Text overlay for key lines. End on strongest visual.<br><br><strong>Posting:</strong> 12pm AEST. Set up ManyChat DM automation — "START" triggers auto-reply with Neuform link.<br><br><strong>Why this works:</strong> Aspirational content on Saturday hits different — people are planning their week. The DM trigger converts viewers while motivation is high. Short duration = high loop rate = algorithm boost.""",
            "examples_cat": "motivation",
            "example_learnings": [
                "Pure motivation with strong visuals. Sometimes the footage speaks for itself. Minimal text, maximum impact — the share happens because people send this to friends who need a push.",
                "Short, punchy motivation shared into DMs. Two words + powerful footage. DM shares are INVISIBLE engagement that Instagram rewards massively.",
                "The 'lock in' energy — showing what commitment looks like in practice, not just theory."
            ],
            "cta": "Comment START &rarr; DM automation with Neuform link"
        }
    ]},
    {"date": "April 5", "day": "Sunday", "week": None, "posts": [
        {
            "platform": "ig", "fmt": "Carousel", "pillar": "workout",
            "title": "My full weekly training split — save this",
            "caption": """My weekly training split.\n\nThis is EXACTLY what I follow on Neuform:\n\nMonday — Push (chest, shoulders, triceps)\nTuesday — Pull (back, biceps)\nWednesday — Legs (quad focus)\nThursday — Active recovery / core\nFriday — Upper body (strength)\nSaturday — Legs (glute/hamstring focus)\nSunday — Rest\n\n10 slides breaking down each day with exercises, sets, and reps.\n\nSave this. Screenshot it. Use it.\n\nOr get the full program with progressions, tracking, and nutrition on Neuform &mdash; link in bio.\n\n#trainingsplit #gym #workout #fitnessprogram #neuform #musclemummy""",
            "options": [
                ("Option B", '"The training split that built this physique" — more visual/physique focused, shows the result of the split'),
                ("Option C", '"Beginner vs Advanced training split" — side by side comparison, broader audience'),
                ("Option D", '"How to build YOUR training split (step by step)" — educational, teaches the WHY')
            ],
            "execution": """<strong>Design:</strong> 10 slides. Neuform brand: Boxer Black background, Papaya accent for day labels, White Inter font for exercise names. One day per slide with 3-4 exercises listed. Final slide = Neuform CTA with app screenshot.<br><br><strong>Photography:</strong> Each slide should have a small exercise image or Chontel performing that day's key lift. B&W photography per brand guidelines with Papaya text overlay.<br><br><strong>Posting:</strong> 7pm Sunday AEST. People plan their week on Sunday night. This is when "save for tomorrow" behaviour peaks. First comment: "What's YOUR split? Drop it below."<br><br><strong>Why this works:</strong> Training split carousels are THE most saved fitness content format on Instagram. Saves signal the algorithm to push to Explore page. Every save is a potential Neuform download — they save the free version, then want the full program.""",
            "examples_cat": "workout",
            "example_learnings": [
                "Clean workout presentation with specific sets and reps. The specificity builds trust — she's not vague, she's giving you the actual program. Chontel should do the same but hold back the progressions for Neuform.",
                "Squat tips format that drove massive saves. The 'save and try at gym' behaviour is the gateway to wanting the full program.",
                "Workout demo with real intensity. The energy and effort visible in the content proves competence better than any caption."
            ],
            "cta": "Save this + full program on Neuform"
        },
        {
            "platform": "tk", "fmt": "Video", "pillar": "nutrition",
            "title": "What I eat in a day — mum of 5 edition",
            "caption": """What I eat in a day.\nMum of 5. Training 6 days a week.\n\n7am — Oats + protein + banana\n10am — Greek yoghurt + berries\n12pm — Chicken + rice + veggies\n3pm — Protein shake + rice cakes\n6pm — Salmon + sweet potato + salad\n8pm — Dark chocolate (non-negotiable)\n\nNo restriction. No guilt. Just fuel.\n\nFull nutrition plans on my app Neuform &mdash; link in IG bio.\n\n#whatieatinaday #fyp #nutrition #fitmum #mealprep #highprotein""",
            "script": {
                "label": "Voiceover Script",
                "hooks": [
                    ("Curiosity", "Everyone asks what I eat with five kids and training six days a week. Here's a full day."),
                    ("Contrarian", "I eat more than most people think. And that's exactly why I look like this."),
                    ("Relatable", "What I eat in a day as a mum who's too busy to meal prep every single thing.")
                ],
                "body": """<em>[quick cuts of each meal being prepared/eaten]</em>\n\nBreakfast — oats with protein powder and banana. Takes two minutes. Kids eat the same thing minus the protein.\n\n<em>[mid-morning snack]</em>\n\nGreek yoghurt and berries between school drop-off and the gym.\n\n<em>[lunch]</em>\n\nChicken, rice, veggies. Nothing fancy. It works.\n\n<em>[afternoon]</em>\n\nProtein shake and rice cakes while the kids have their snack.\n\n<em>[dinner]</em>\n\nSalmon, sweet potato, big salad. The whole family eats this one.\n\n<em>[evening]</em>\n\nAnd dark chocolate. Every night. Non-negotiable.\n\nNo restriction. No guilt. Just fuel.\n\nAll my meal plans are on Neuform — link in my IG bio."""
            },
            "options": [
                ("Option B", '"Eating 2200 calories as a training mum — here\'s what that actually looks like" — specific calories/macros angle'),
                ("Option C", '"Meals my kids will actually eat that still hit my macros" — family-friendly angle')
            ],
            "execution": """<strong>Filming:</strong> Film throughout the actual day — each meal as you eat it. Phone propped up, quick clips. The food should look real and homemade, not Instagram-perfect. Brand guidelines: authentic, genuine.<br><br><strong>Editing:</strong> 30-45 seconds. Quick cuts. Text overlay with meal name + macros. Trending sound bed under voiceover OR just voiceover.<br><br><strong>Posting:</strong> 12pm AEST Sunday. WIEIAD content performs best midday when people are thinking about food. Use #whatieatinaday — it has 15B+ views on TikTok.<br><br><strong>Why this works:</strong> WIEIAD is consistently TikTok's highest-performing fitness format. The mum-of-5 angle makes it unique. The 'non-negotiable chocolate' humanises Chontel and fights the restrictive fitness stereotype. Natural Neuform nutrition plan CTA.""",
            "examples_cat": "nutrition",
            "example_learnings": [
                "Simple high-protein recipe with specific numbers. The SPECIFICITY (exact grams, exact calories) builds trust. Chontel should always include macros — it signals expertise and drives saves.",
                "Full day of eating with realistic portions. Not performative — real food a real person eats. This is what Chontel's audience wants to see.",
                "Budget-friendly meal prep that's still high protein. Practical value = saves. Saves = algorithm fuel = more reach = more app downloads."
            ],
            "cta": "Full meal plans on Neuform app"
        }
    ]},

    # === WEEK 2 ===
    {"date": "April 6", "day": "Monday", "week": "Week 2 &mdash; April 6-12 &mdash; Building Momentum", "posts": [
        {
            "platform": "ig", "fmt": "Reel", "pillar": "workout",
            "title": "Monday push session — full upper body",
            "caption": """Monday. Let's go.\n\nToday's push session:\n1. Incline DB press — 4x8\n2. Cable flyes — 3x12\n3. Seated shoulder press — 4x10\n4. Lateral raises — 3x15\n5. Tricep pushdowns — 3x12\n\nThis took me 45 minutes. That's it.\n\nYou don't need two hours. You need intensity and a plan.\n\nFull program with weekly progressions on Neuform &mdash; link in bio.\n\n#mondaymotivation #upperbody #pushday #gym #neuform""",
            "script": {
                "label": "Coaching Voiceover",
                "hooks": [
                    ("Energy", "It's Monday. New week. No excuses. Let's get this upper body session done."),
                    ("Value", "Forty-five minutes. Five exercises. That's all you need for a complete push session. Let me show you."),
                    ("Challenge", "If you can't give me forty-five minutes on a Monday, we need to talk.")
                ],
                "body": """<em>[footage: walking into gym, setting up]</em>\n\nFirst up — incline dumbbell press. Four sets of eight. Go heavy. Control the negative.\n\n<em>[show exercise]</em>\n\nCable flyes next. Squeeze at the top. Feel the chest working.\n\n<em>[show exercise]</em>\n\nSeated shoulder press. Core tight. Don't arch your back.\n\n<em>[show exercise]</em>\n\nLateral raises — lighter weight, higher reps. Burn is the goal.\n\n<em>[show exercise]</em>\n\nFinish with tricep pushdowns. Lock out every rep.\n\n<em>[post-workout]</em>\n\nForty-five minutes. Done. Full program on Neuform — link in bio."""
            },
            "options": [
                ("Option B", '"The upper body workout busy mums can do in 30 minutes" — shorter, more accessible'),
                ("Option C", '"Upper body mistakes I see every day at the gym" — educational/correction angle')
            ],
            "execution": """<strong>Filming:</strong> Film the ACTUAL session. One angle per exercise, good lighting, gym background. Show real weight on the bar — credibility matters.<br><br><strong>Editing:</strong> 30-45 seconds. One exercise flows into the next. Text overlay: exercise name + sets x reps. Voiceover with coaching cues.<br><br><strong>Posting:</strong> 6:30am AEST Monday. "Monday motivation" is the most searched fitness term at the start of the week. First comment: "Save this for your next push day."<br><br><strong>Why this works:</strong> Structured workout content with specific sets/reps drives saves. People screenshot these to use at the gym. Every save-and-use moment is a "I need her full program" moment.""",
            "examples_cat": "workout",
            "example_learnings": [
                "Full workout with real coaching energy. The intensity visible in the content proves competence.",
                "Clean demonstration with specific rep counts. Specificity drives saves — people screenshot this for the gym.",
                "Form-focused content that establishes coaching authority. This is the trust-building that makes someone download an app."
            ],
            "cta": "Link in bio &rarr; Neuform full program"
        },
        {
            "platform": "tk", "fmt": "Video", "pillar": "workout",
            "title": "The back workout that changed my physique",
            "caption": """The back workout that changed EVERYTHING.\n\nI ignored my back for years. Big mistake.\n\nNow it's my strongest muscle group:\n&bull; Barbell rows — heavy, strict\n&bull; Lat pulldowns — full stretch\n&bull; Single arm DB rows — squeeze at top\n&bull; Face pulls — every session\n\nA strong back changes your entire silhouette.\n\nFollow for more + full program link in bio.\n\n#backmuscles #fyp #gym #musclemummy #workout""",
            "script": {
                "label": "Voiceover Script",
                "hooks": [
                    ("Story", "I used to skip back day. For years. Then I realised it was the reason my physique looked unbalanced."),
                    ("Contrarian", "Everyone's obsessed with glutes. But the muscle that actually changed how I look? My back."),
                    ("Visual", "See this back? It didn't look like this two years ago. Here's exactly what I did.")
                ],
                "body": """<em>[start with a back flex or rear shot]</em>\n\nThis back didn't happen by accident. Four exercises. Twice a week. Consistency.\n\nBarbell rows — go heavy. Strict form. Feel your lats pull.\n\n<em>[show exercise]</em>\n\nLat pulldowns — full stretch at the top, squeeze at the bottom.\n\n<em>[show exercise]</em>\n\nSingle arm rows — this is where the detail comes from.\n\n<em>[show exercise]</em>\n\nFace pulls — I finish every upper body session with these. Non-negotiable for posture and rear delts.\n\n<em>[final back shot]</em>\n\nTrain your back. Trust me."""
            },
            "options": [
                ("Option B", '"Back exercises ranked from overrated to underrated" — tier list format, drives comments'),
                ("Option C", '"Why every woman should train back" — educational, broader audience appeal')
            ],
            "execution": """<strong>Filming:</strong> Chontel is known for her back development — showcase it. Film from behind for the flex shots. B&W per brand guidelines for the physique shots. Colour for the exercise demos.<br><br><strong>Editing:</strong> 20-30 seconds. Strong opening visual (the back). Quick cuts between exercises.<br><br><strong>Why this works:</strong> Physique-specific content performs well because people search body-part keywords. "Back workout" is a high-volume search term on TikTok. This positions Chontel's physique as the proof.""",
            "examples_cat": "workout",
            "example_learnings": [
                "Physique-focused content where the body IS the proof. No need to oversell — the visual does the work.",
                "Exercise-specific content that people save for their next gym session. Saves = algorithm priority.",
                "Strong woman training hard with minimal caption. Let the strength speak."
            ],
            "cta": "Follow + link in bio for full program"
        },
        {
            "platform": "story", "fmt": "App Demo", "pillar": "app",
            "title": "Weekly Neuform app walkthrough",
            "caption": """Slide 1: "Every Monday I'll show you inside my app Neuform"\nSlide 2: Screen record — opening the app, browsing programs\nSlide 3: Show selecting today's workout\nSlide 4: Show the nutrition plan feature\nSlide 5: "Try it free for 7 days — link right here" + link sticker""",
            "script": {
                "label": "Talking-to-Camera Script",
                "hooks": [
                    ("Helpful", "A lot of you have been asking what Neuform actually looks like inside. Let me show you."),
                    ("Personal", "This is the app I built because I couldn't find a program that worked for busy mums. Let me walk you through it."),
                    ("Social proof", "Over a thousand women are using this right now. Here's what they see when they open the app.")
                ],
                "body": """<em>[screen recording of Neuform app]</em>\n\nSo when you open Neuform, you pick your program. I've got programs for beginners, intermediate, advanced — and specific postpartum programs.\n\n<em>[tap through to a workout]</em>\n\nHere's today's session. Every exercise has a video demo, sets, reps, and rest times. You just follow along.\n\n<em>[show nutrition section]</em>\n\nAnd this is the nutrition side. Full meal plans, shopping lists, macro breakdowns. All customisable.\n\n<em>[back to camera]</em>\n\nSeven days free. No commitment. Link's right here. Try it."""
            },
            "options": [
                ("Option B", '"3 features in Neuform you didn\'t know about" — highlights lesser-known features'),
            ],
            "execution": """<strong>Format:</strong> Screen record the app with voiceover. Keep it under 60 seconds across 4-5 Story slides. This should feel like a friend showing you an app, not a sales pitch.<br><br><strong>Posting:</strong> Every Monday. Consistency builds expectation — followers learn "Monday = app walkthrough." This is the lowest-friction way to showcase the product.<br><br><strong>Why this works:</strong> Most people don't download an app because they don't know what's inside. Showing the actual experience removes uncertainty. The 7-day free trial removes financial risk. Combined = highest conversion rate.""",
            "examples_cat": "transformation",
            "example_learnings": [
                "App credited directly in transformation content. 'Thanks [app name]' — making the product the hero. Chontel should do this naturally in every app walkthrough.",
                "Challenge launch tied to app download. The app isn't sold as a product — it's sold as the TOOL for the challenge.",
                "Community-driven content where the app is the gathering point. Position Neuform as where the community lives."
            ],
            "cta": "Try Neuform free for 7 days &mdash; link sticker"
        }
    ]},
]

# Continue generating remaining days...
# I'll create a more complete dataset

more_days = [
    # Apr 7
    {"date": "April 7", "day": "Tuesday", "week": None, "posts": [
        {"platform": "ig", "fmt": "Carousel", "pillar": "transformation", "title": "Client spotlight: 8-week transformation",
         "caption": "Meet Sarah.\n\n8 weeks ago she messaged me saying she hadn't trained since having her second baby.\n\nShe was nervous. Unsure. Didn't know where to start.\n\nI said: \"Start with Neuform. Follow the program. Trust the process.\"\n\n8 weeks later &mdash; this.\n\nSame woman. Same life. Same responsibilities.\nDifferent energy. Different confidence. Different body.\n\nThis is what structure does.\n\nDM me TRANSFORM to get started.\n\n#transformation #beforeandafter #neuform #fitnessprogram #fitmum",
         "options": [("Option B", '"3 client transformations that made ME emotional" — multiple in one post, more social proof'), ("Option C", '"What she did differently this time (it wasn\'t a crash diet)" — educational angle on why programs work')],
         "execution": "<strong>Design:</strong> Before photo (slide 1), journey slides (2-4), after photo (slide 5), her quote (slide 6), program she followed (slide 7), Neuform CTA (slide 8). Use B&W for the photos per brand guidelines. Papaya accent for text.<br><br><strong>Important:</strong> ALWAYS get written permission. Feature her first name only unless she wants full name. Tag her if she has an account.<br><br><strong>Why this works:</strong> Client spotlights are the #1 conversion driver for fitness apps. The viewer thinks 'she's like me — if it worked for her, it'll work for me.' DM trigger CTA captures high-intent leads.",
         "examples_cat": "transformation",
         "example_learnings": ["Transformation that emphasises PROCESS over quick fixes. Caption talks about consistency, not shortcuts.", "App credited in the transformation result. The program is the hero.", "Before/after that shows confidence change, not just physical change. This resonates deeper."],
         "cta": "DM me TRANSFORM to get started"},
        {"platform": "tk", "fmt": "Video", "pillar": "mumlife", "title": "Things only gym mums understand",
         "caption": "Things only gym mums understand:\n\n&bull; Packing a gym bag AND a nappy bag\n&bull; Doing bicep curls with a toddler on your hip\n&bull; Your pre-workout is cold coffee from 6am\n&bull; \"Rest day\" means you only carried one child\n&bull; Finding a Lego in your sports bra\n\nIf you know, you know.\n\nTag your gym mum bestie.\n\n#fyp #fitmum #gymhumor #mumlife #musclemummy #relatable",
         "script": {"label": "Voiceover Script", "hooks": [("Relatable", "If you're a mum who trains — you're going to feel this in your soul."), ("Humour", "Things absolutely nobody warned me about when I became a gym mum."), ("Community", "This one's for the mums who train with Legos in their shoes and yesterday's coffee as pre-workout.")],
          "body": "<em>[act out each scenario with quick cuts]</em>\n\nPacking a gym bag AND a nappy bag. <em>[show both bags]</em>\n\nDoing bicep curls with a toddler on your hip. <em>[actually do it, or mime it]</em>\n\nYour pre-workout is cold coffee from six am. <em>[hold up sad coffee]</em>\n\n\"Rest day\" just means you only carried one child instead of three. <em>[exhausted face]</em>\n\nFinding a Lego in your sports bra mid-set. <em>[pull out Lego, dead stare to camera]</em>\n\nIf you know... you know.\n\nTag your gym mum bestie."},
         "options": [("Option B", '"A day in the life of a gym mum — expectation vs reality" — split screen comparison'), ("Option C", '"Rating my gym mum hacks from genius to unhinged" — self-deprecating humour tier list')],
         "execution": "<strong>Filming:</strong> Quick skits. Each scenario = 3-4 seconds. Props: gym bag, nappy bag, cold coffee, Lego. Can film at home AND gym. Kids can cameo — makes it real.<br><br><strong>Editing:</strong> 15-20 seconds. Fast cuts. Trending funny sound. Text overlay for each point. End on the Lego moment (strongest punchline last).<br><br><strong>Why this works:</strong> Humour drives the highest SHARE rate on TikTok. Every share goes to another mum who goes 'OH MY GOD that's me.' This is how you reach mums who DON'T follow fitness accounts yet — through their friends' shares.",
         "examples_cat": "mumlife",
         "example_learnings": ["Mum fitness content with personality. Not just workouts — the LIFE around the workouts.", "Relatable mum struggle that shows authenticity. This builds the 'she's one of us' connection.", "Fun energy that makes fitness feel approachable, not intimidating."],
         "cta": "Tag your gym mum bestie"},
        {"platform": "story", "fmt": "Q&A", "pillar": "motivation", "title": "Motivational quote + voice note + question box",
         "caption": "Slide 1: Quote card — \"The woman who shows up on the hard days is the one who changes her life.\"\nSlide 2: Voice note — why you started training\nSlide 3: Question box: \"What's YOUR why?\"",
         "script": {"label": "Voice Note Script", "hooks": [("Reflective", "I've been thinking about WHY I train. It's not what most people think."), ("Honest", "Someone asked me yesterday why I still train after all these years. And my answer surprised even me."), ("Inspiring", "I want to talk about the real reason I show up every day — and it has nothing to do with how I look.")],
          "body": "I started training because I wanted to look good. That's the honest truth.\n\nBut the reason I KEPT training — after five babies, after injuries, after the days I wanted to quit — is because of how it makes me FEEL.\n\nI feel like myself when I train. I feel strong. I feel capable. I feel like a good mum.\n\nAnd I want every woman who follows me to have that feeling too.\n\nThat's why I built Neuform. That's why I'm here.\n\nSo tell me — what's YOUR why? Drop it in the question box."},
         "options": [],
         "execution": "<strong>Format:</strong> 3 slides. Quote card designed in Neuform brand colours (Boxer Black + Papaya text). Voice note should be raw — record once, don't re-record. Imperfection is authenticity.<br><br><strong>Why this works:</strong> Question boxes generate direct replies which Instagram treats as high-value engagement. Every reply = conversation = higher Story placement next time. The voice note creates intimacy that text can't.",
         "examples_cat": "motivation",
         "example_learnings": ["Emotional vulnerability that builds deep connection.", "Motivation that comes from real experience, not generic quotes."],
         "cta": "Question box: What's YOUR why?"}
    ]},
    # Apr 8
    {"date": "April 8", "day": "Wednesday", "week": None, "posts": [
        {"platform": "ig", "fmt": "Reel", "pillar": "nutrition", "title": "3 high-protein meals I eat every single week",
         "caption": "3 meals on repeat every single week:\n\n1. Chicken stir-fry — 42g protein\nChicken thigh + rice + whatever veggies are in the fridge. Soy sauce, garlic, done. 10 minutes.\n\n2. Protein overnight oats — 35g protein\nOats + protein powder + Greek yoghurt + milk. Make it the night before. Grab and go.\n\n3. Tuna rice bowls — 38g protein\nTinned tuna + brown rice + avocado + cucumber. Sounds boring. Tastes incredible with the right seasoning.\n\nSimple. Repeatable. High protein.\n\nAll my recipes and full meal plans are on Neuform — link in bio.\n\n#highprotein #mealprep #nutrition #healthyrecipes #neuform",
         "script": {"label": "Voiceover Script", "hooks": [("Value", "Three meals. Every single week. Over thirty-five grams of protein each. And they take less than ten minutes."), ("Honest", "I don't eat fancy. I eat the same three meals every week and it works. Here they are."), ("Relatable", "I've got five kids. I don't have time for complicated recipes. These are my three go-to meals.")],
          "body": "<em>[quick cuts of each meal being made]</em>\n\nMeal one — chicken stir-fry. Forty-two grams of protein. Chicken thigh, rice, whatever veggies you've got. Soy sauce, garlic. Ten minutes.\n\n<em>[show plating up]</em>\n\nMeal two — protein overnight oats. Thirty-five grams. Mix it the night before. Grab it in the morning. Done.\n\n<em>[show prepping the night before]</em>\n\nMeal three — tuna rice bowl. Thirty-eight grams of protein. Tinned tuna, brown rice, avocado. Season it properly and it's genuinely delicious.\n\n<em>[show finished bowl]</em>\n\nSimple. Repeatable. Works every time.\n\nAll my recipes and full meal plans are on Neuform. Link in bio."},
         "options": [("Option B", '"What I\'d eat if I had $50 for the whole week" — budget angle'), ("Option C", '"High protein meals my kids actually eat too" — family crossover')],
         "execution": "<strong>Filming:</strong> Top-down and side angle of food prep. Natural kitchen lighting. Quick cuts. The food should look REAL and homemade, not styled.<br><br><strong>Posting:</strong> 12pm AEST. Recipe Reels perform best at lunchtime when people are thinking about food.<br><br><strong>Why this works:</strong> Recipe Reels have the highest save rate of any fitness content. Saves signal algorithm to push to Explore. The mum angle ('five kids, no time') differentiates this from the thousand other protein meal posts.",
         "examples_cat": "nutrition",
         "example_learnings": ["Simple recipe with specific protein grams. Specificity builds trust and drives saves.", "Food that looks real, not styled. Authenticity > aesthetics for this audience.", "Practical value content that people save and actually use. Every save = algorithm fuel."],
         "cta": "All recipes on Neuform &mdash; link in bio"},
        {"platform": "tk", "fmt": "Video", "pillar": "workout", "title": "Gym exercises ranked: overrated to underrated",
         "caption": "Gym exercises ranked:\n\nOVERRATED:\n&bull; Leg press (too easy to ego lift)\n&bull; Crunches (do ANYTHING else)\n&bull; Smith machine squats (fight me)\n\nUNDERRATED:\n&bull; Romanian deadlifts\n&bull; Face pulls\n&bull; Hip thrust with pause\n\nDrop your hot take in the comments.\n\n#fyp #gymtok #exerciseranking #fitness #workout",
         "script": {"label": "Voiceover Script", "hooks": [("Contrarian", "I'm about to make some enemies. Let's rank gym exercises from overrated to underrated."), ("Challenge", "Tell me your most overrated exercise and I'll tell you mine. Here's my list."), ("Authority", "Thirteen years of coaching. These are the exercises you're wasting your time on.")],
          "body": "<em>[tier list graphic or just text overlays as you talk]</em>\n\nOverrated. Leg press. Too easy to load up plates and barely move. Your ego gets a workout. Your legs don't.\n\nOverrated. Crunches. Do literally anything else for your core. Please.\n\nOverrated. Smith machine squats. I said what I said.\n\n<em>[transition]</em>\n\nNow the underrated ones.\n\nRomanian deadlifts. If you're not doing these, you're leaving hamstring and glute gains on the table.\n\nFace pulls. I finish every upper body session with these. Your posture will thank you.\n\nHip thrust with a PAUSE at the top. Three second hold. You'll feel this for days.\n\nDrop your hot take below. I want to hear it."},
         "options": [("Option B", '"Exercises I stopped doing and what I replaced them with" — before/after format'), ("Option C", '"The exercise I hated that ended up changing my body" — story format')],
         "execution": "<strong>Filming:</strong> Can be direct-to-camera with text overlays OR clips of each exercise. The ranking/tier list format is native to TikTok and drives MASSIVE comments because people disagree.<br><br><strong>Editing:</strong> 20-30 seconds. Quick cuts between exercises. Strong opinions stated confidently.<br><br><strong>Why this works:</strong> Opinion/ranking content drives the highest COMMENT rate on TikTok. Comments = algorithm fuel. People will argue in the comments for days. Every comment pushes the video to more people. Chontel gets to show expertise AND personality.",
         "examples_cat": "workout",
         "example_learnings": ["Strong opinions drive engagement. Don't be generic — be polarising.", "Quick exercise demos with coaching cues. Authority comes from specificity.", "Content that invites debate = comments = algorithm rocket fuel."],
         "cta": "Drop your hot take in the comments"}
    ]},
]

days.extend(more_days)

# Generate remaining days (Apr 9-30) with the content plan
remaining = [
    ("April 9", "Thursday", None, [
        ("ig", "Reel", "workout", "The leg workout that will humble you", True),
        ("tk", "Video", "transformation", "I had 5 C-sections. Here's my core now.", True),
        ("story", "Behind Scenes", "nutrition", "What's in my gym bag + what's in my fridge", False)
    ]),
    ("April 10", "Friday", None, [
        ("ig", "Carousel", "motivation", "5 things I wish I knew before starting fitness", False),
        ("tk", "Video", "workout", "Follow along: 10 min HIIT anywhere", True),
        ("story", "App Demo", "app", "Show a member comment + app demo", False)
    ]),
    ("April 11", "Saturday", None, [
        ("ig", "Reel", "mumlife", "What my partner thinks I do vs what I actually do", True),
        ("tk", "Video", "nutrition", "Healthy snacks my kids actually eat", True),
    ]),
    ("April 12", "Sunday", None, [
        ("ig", "Reel", "motivation", "Sunday reminder: one workout away from a better mood", False),
        ("tk", "Video", "mumlife", "Sunday reset as a mum of 5", True),
    ]),
    ("April 13", "Monday", "Week 3 &mdash; April 13-19 &mdash; NEW PROGRAM LAUNCH (Key Conversion Week)", [
        ("ig", "Reel", "workout", "NEW PROGRAM DROP: April program LIVE on Neuform", True),
        ("tk", "Video", "workout", "Day 1 of my new program", True),
        ("story", "Launch", "transformation", "Countdown reveal + testimonials + launch hype", True)
    ]),
    ("April 14", "Tuesday", None, [
        ("ig", "Carousel", "workout", "My April program breakdown — Week 1-4 overview", False),
        ("tk", "Video", "transformation", "Reacting to my body 1 year postpartum vs now", True),
        ("story", "Behind Scenes", "mumlife", "BTS: juggling filming content + school pickup", False)
    ]),
    ("April 15", "Wednesday", None, [
        ("ig", "Reel", "nutrition", "Post-workout meal in under 10 minutes", True),
        ("tk", "Video", "motivation", "She said mums can't be muscular", True),
        ("story", "Interactive", "workout", "Live Q&A: form check — followers submit videos", False)
    ]),
    ("April 16", "Thursday", None, [
        ("ig", "Reel", "mumlife", "The 5am alarm hits different with 5 kids", True),
        ("tk", "Video", "workout", "3 glute exercises you're NOT doing", True),
        ("story", "Social Proof", "transformation", "Share 3 member results from week 1 of new program", False)
    ]),
    ("April 17", "Friday", None, [
        ("ig", "Carousel", "workout", "Upper body at home — no equipment needed", False),
        ("tk", "Video", "nutrition", "Eating 2200 calories as a training mum", True),
        ("story", "Community", "motivation", "Friday wins roundup — share your accomplishments", False)
    ]),
    ("April 18", "Saturday", None, [
        ("ig", "Reel", "couple", "Partner workout with my husband", True),
        ("tk", "Video", "mumlife", "POV: Your mum is a bodybuilder", True),
    ]),
    ("April 19", "Sunday", None, [
        ("ig", "Carousel", "nutrition", "Sunday meal prep — feeds a family of 7", False),
        ("tk", "Video", "motivation", "This time last year vs now", False),
    ]),
    ("April 20", "Monday", "Week 4 &mdash; April 20-26 &mdash; Results &amp; Social Proof", [
        ("ig", "Reel", "workout", "The workout that changed my back", True),
        ("tk", "Video", "workout", "Exercises I'll never stop doing — top 5", True),
        ("story", "App Demo", "app", "Neuform feature: meal plan customisation", True)
    ]),
    ("April 21", "Tuesday", None, [
        ("ig", "Reel", "transformation", "From postpartum to pulling 100kg", True),
        ("tk", "Video", "mumlife", "Gym bro vs gym mum — side by side", True),
        ("story", "Day in Life", "mumlife", "School run → gym → HIIT Capalaba → family dinner", False)
    ]),
    ("April 22", "Wednesday", None, [
        ("ig", "Carousel", "motivation", "Mindset shifts that changed my fitness journey", False),
        ("tk", "Video", "nutrition", "Stop eating this if you want to lose fat", True),
        ("story", "Educational", "workout", "Quick form tutorial: how to hip hinge properly", False)
    ]),
    ("April 23", "Thursday", None, [
        ("ig", "Reel", "workout", "The shoulder workout I swear by", True),
        ("tk", "Video", "mumlife", "What my kids think I do for work", True),
        ("story", "Social Proof", "transformation", "Mid-month check-in: member April program progress", False)
    ]),
    ("April 24", "Friday", None, [
        ("ig", "Reel", "mumlife", "Friday night as a fit mum — not what you'd expect", True),
        ("tk", "Video", "workout", "1 exercise to fix your posture", True),
        ("story", "Interactive", "nutrition", "Weekend treat meal ideas — poll: pizza or burger?", False)
    ]),
    ("April 25", "Saturday", None, [
        ("ig", "Reel", "motivation", "To every mum who thinks it's too late — watch this", True),
        ("tk", "Video", "workout", "Full body workout in my garage", True),
    ]),
    ("April 26", "Sunday", None, [
        ("ig", "Carousel", "workout", "My top 10 exercises of all time — ranked", False),
        ("tk", "Video", "nutrition", "Protein sources I eat daily", True),
    ]),
    ("April 27", "Monday", "Week 5 &mdash; April 27-30 &mdash; Close the Month Strong (Final Conversion Push)", [
        ("ig", "Reel", "workout", "New week, new energy — Monday push session", True),
        ("tk", "Video", "motivation", "30 seconds of proof that consistency beats everything", False),
        ("story", "App Demo", "app", "Feature spotlight: progress tracking in Neuform", True)
    ]),
    ("April 28", "Tuesday", None, [
        ("ig", "Carousel", "transformation", "April challenge results so far — member before/afters", False),
        ("tk", "Video", "workout", "The exercise I avoided for years (and why I was wrong)", True),
        ("story", "Vulnerable", "mumlife", "Real talk: mum guilt about training time", True)
    ]),
    ("April 29", "Wednesday", None, [
        ("ig", "Reel", "nutrition", "My grocery haul for a family of 7 — all healthy", True),
        ("tk", "Video", "mumlife", "Things I heard when pregnant and still training", True),
        ("story", "Community", "motivation", "Month-end reflection: What did YOU achieve in April?", False)
    ]),
    ("April 30", "Thursday", None, [
        ("ig", "Reel", "motivation", "April wrap-up: What a month. Here's what we achieved.", True),
        ("tk", "Video", "transformation", "30 days of consistency — watch what happens", False),
        ("story", "Urgency", "app", "Last chance: 7-day free trial this month — urgency CTA", True)
    ]),
]

# Now I'll save the full plan as a structured JSON that the main script can use
plan = {"detailed_days": days, "remaining_days": remaining}
with open('/tmp/full_calendar_plan.json', 'w') as f:
    json.dump(plan, f, indent=2, default=str)

print(f"Detailed days (with full content): {len(days)}")
print(f"Remaining days (to be generated): {len(remaining)}")
print(f"Total: {len(days) + len(remaining)} days")
print(f"\nDetailed posts ready: {sum(len(d['posts']) for d in days)}")
print(f"Remaining posts to generate: {sum(len(d[3]) for d in remaining)}")
