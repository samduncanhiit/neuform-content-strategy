"""Pins the bot to MindBody-only scope: tools, handlers, routes, prompt."""
import contextlib
import inspect
import os
import unittest
from unittest.mock import patch

import app
import mindbody_helper

EXPECTED_TOOLS = {
    "get_todays_classes", "get_daily_briefing", "search_clients",
    "get_client_detail", "get_member_stats", "get_payment_failures",
    "get_classes_history", "get_revenue", "get_new_members",
    "get_membership_movement", "get_arrears_report", "get_weekly_summary",
    "run_class_report", "get_noshow_report",
}

REMOVED_TOOLS = [
    "get_trello_tasks", "add_trello_card", "edit_trello_card",
    "move_trello_card", "remove_trello_card", "get_jotform_submissions",
    "get_calendar_events", "create_calendar_event", "read_inbox",
    "draft_email", "read_gmail", "read_gmail_drafts",
]

REMOVED_MODULES = [
    "gcal_helper", "gmail_helper", "outlook_helper",
    "trello_helper", "jotform_helper", "lead_automation",
]

ERIN = "+61421188443"
SAM = "+61420233508"

APP_SOURCE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@contextlib.contextmanager
def stubbed_mindbody():
    """Replace every public mindbody_helper function with a stub returning '<name>'."""
    with contextlib.ExitStack() as stack:
        for name, obj in list(vars(mindbody_helper).items()):
            if inspect.isfunction(obj) and not name.startswith("_"):
                stack.enter_context(
                    patch.object(mindbody_helper, name, return_value=f"<{name}>")
                )
        yield


class TestToolSurface(unittest.TestCase):
    def test_tool_list_is_exactly_mindbody(self):
        names = {t["name"] for t in app._MINDBODY_TOOLS}
        self.assertEqual(names, EXPECTED_TOOLS)

    def test_every_tool_has_a_handler(self):
        with stubbed_mindbody():
            for name in sorted(EXPECTED_TOOLS):
                with self.subTest(tool=name):
                    result = app.handle_tool_call(name, {})
                    self.assertFalse(result.startswith("Unknown tool"), result)

    def test_removed_tools_are_unknown(self):
        for name in REMOVED_TOOLS:
            with self.subTest(tool=name):
                self.assertEqual(app.handle_tool_call(name, {}), f"Unknown tool: {name}")

    def test_handle_tool_call_takes_only_name_and_input(self):
        params = list(inspect.signature(app.handle_tool_call).parameters)
        self.assertEqual(params, ["tool_name", "tool_input"])

    def test_daily_briefing_is_classes_only(self):
        with stubbed_mindbody():
            self.assertEqual(app.handle_tool_call("get_daily_briefing", {}), "<format_briefing>")

    def test_daily_briefing_description_has_no_calendar_or_inbox(self):
        tool = next(t for t in app._MINDBODY_TOOLS if t["name"] == "get_daily_briefing")
        self.assertNotIn("calendar", tool["description"].lower())
        self.assertNotIn("inbox", tool["description"].lower())


class TestAppSurface(unittest.TestCase):
    def test_only_mindbody_routes_remain(self):
        rules = {r.rule for r in app.app.url_map.iter_rules() if r.endpoint != "static"}
        self.assertEqual(rules, {"/webhook", "/health", "/"})

    def test_lead_scheduler_removed(self):
        self.assertFalse(hasattr(app, "start_scheduler"))
        self.assertFalse(hasattr(app, "_run_daily_leads"))

    def test_per_user_integration_config_removed(self):
        for attr in ("USER_EMAILS", "USER_GMAIL", "USER_CALENDAR", "USER_TRELLO",
                     "_get_tools_for_user", "ALL_TOOLS"):
            with self.subTest(attr=attr):
                self.assertFalse(hasattr(app, attr))

    def test_slow_keywords_have_no_removed_features(self):
        for kw in ("trello", "hiit challenge", "tasks", "submission",
                   "submissions", "jotform", "form submissions"):
            with self.subTest(kw=kw):
                self.assertNotIn(kw, app.SLOW_KEYWORDS)

    def test_no_removed_module_referenced_in_app(self):
        with open(APP_SOURCE_PATH) as f:
            source = f.read()
        for mod in REMOVED_MODULES:
            with self.subTest(module=mod):
                self.assertNotIn(mod, source)


class TestSystemPrompt(unittest.TestCase):
    def test_prompt_declares_mindbody_only_scope(self):
        prompt = app._build_system_prompt("Sam", SAM)
        self.assertIn("this bot only handles MindBody questions", prompt)

    def test_prompt_has_no_removed_tool_guidance(self):
        prompt = app._build_system_prompt("Sam", SAM)
        for phrase in ("get_calendar_events", "read_inbox", "read_gmail",
                       "add_trello_card", "move_trello_card", "remove_trello_card",
                       "get_jotform_submissions", "create a draft",
                       "Prefix event name", "two email accounts"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, prompt)

    def test_prompt_keeps_mindbody_routing_rules(self):
        prompt = app._build_system_prompt("Sam", SAM)
        for phrase in ("CRITICAL RULE — verbatim tool output", "get_membership_movement",
                       "get_client_detail", "get_noshow_report", "FORMATTING:"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, prompt)


class TestRevenuePassword(unittest.TestCase):
    def _prompt(self, raw_number, env):
        with patch.dict(os.environ, env, clear=False):
            if "REVENUE_PASSWORD" not in env:
                os.environ.pop("REVENUE_PASSWORD", None)
            return app._build_system_prompt("Erin" if raw_number == ERIN else "Sam", raw_number)

    def test_password_comes_from_env(self):
        prompt = self._prompt(ERIN, {"REVENUE_PASSWORD": "hunter2"})
        self.assertIn("Correct password: 'hunter2'", prompt)

    def test_password_is_stripped(self):
        prompt = self._prompt(ERIN, {"REVENUE_PASSWORD": "  hunter2\n"})
        self.assertIn("Correct password: 'hunter2'", prompt)

    def test_unset_password_fails_closed(self):
        prompt = self._prompt(ERIN, {})
        self.assertNotIn("Correct password", prompt)
        self.assertIn("Never call get_revenue for this user", prompt)

    def test_whitespace_password_fails_closed(self):
        prompt = self._prompt(ERIN, {"REVENUE_PASSWORD": "   "})
        self.assertNotIn("Correct password", prompt)
        self.assertIn("Never call get_revenue for this user", prompt)

    def test_other_users_never_see_password_text(self):
        prompt = self._prompt(SAM, {"REVENUE_PASSWORD": "hunter2"})
        self.assertNotIn("hunter2", prompt)
        self.assertNotIn("get_revenue for this user", prompt)

    def test_old_password_not_in_source(self):
        with open(APP_SOURCE_PATH) as f:
            self.assertNotIn("samistheman", f.read())


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestRepoContents(unittest.TestCase):
    def test_removed_modules_are_gone(self):
        for mod in REMOVED_MODULES:
            with self.subTest(module=mod):
                self.assertFalse(os.path.exists(os.path.join(REPO_ROOT, f"{mod}.py")))

    def test_content_calendar_is_gone(self):
        self.assertFalse(os.path.exists(os.path.join(REPO_ROOT, "content-calendar")))

    def test_mindbody_maintenance_script_kept(self):
        self.assertTrue(os.path.exists(os.path.join(REPO_ROOT, "docs", "list_contracts.py")))

    def test_requirements_are_mindbody_only(self):
        with open(os.path.join(REPO_ROOT, "requirements.txt")) as f:
            packages = {line.split("==")[0].strip().lower()
                        for line in f if line.strip() and not line.startswith("#")}
        self.assertEqual(packages, {"requests", "flask", "twilio", "anthropic", "gunicorn"})


if __name__ == "__main__":
    unittest.main()
