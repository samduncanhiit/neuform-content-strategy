"""Unit tests for the membership movement report pure functions."""
import unittest

import mindbody_helper


class TestScaffolding(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(hasattr(mindbody_helper, "_is_tracked_membership"))


class TestBucketByMonth(unittest.TestCase):
    def _event(self, date, name="X", membership="All Access 6 Month"):
        return {"client_id": 1, "contract_id": 1, "name": name,
                "membership": membership, "date": date}

    def test_events_grouped_by_calendar_month(self):
        signups = [
            self._event("2026-02-14", "Jane"),
            self._event("2026-03-02", "Bob"),
            self._event("2026-03-30", "Carol"),
        ]
        cancellations = [self._event("2026-02-20", "Dan")]
        buckets = mindbody_helper._bucket_movement_by_month(
            signups, cancellations,
            window_start="2026-01-16", window_end="2026-04-16",
        )
        labels = [b["month_label"] for b in buckets]
        self.assertEqual(labels,
            ["January 2026", "February 2026", "March 2026", "April 2026"])
        feb = buckets[1]
        self.assertEqual(len(feb["signups"]), 1)
        self.assertEqual(feb["signups"][0]["name"], "Jane")
        self.assertEqual(len(feb["cancellations"]), 1)
        self.assertEqual(len(buckets[2]["signups"]), 2)
        self.assertEqual(buckets[0]["signups"], [])
        self.assertEqual(buckets[3]["signups"], [])

    def test_partial_flag_on_edge_months(self):
        buckets = mindbody_helper._bucket_movement_by_month(
            signups=[], cancellations=[],
            window_start="2026-01-16", window_end="2026-04-16",
        )
        partial_map = {b["month_label"]: b["partial"] for b in buckets}
        self.assertTrue(partial_map["January 2026"])
        self.assertFalse(partial_map["February 2026"])
        self.assertFalse(partial_map["March 2026"])
        self.assertTrue(partial_map["April 2026"])

    def test_full_calendar_window_has_no_partials(self):
        buckets = mindbody_helper._bucket_movement_by_month(
            signups=[], cancellations=[],
            window_start="2026-02-01", window_end="2026-03-31",
        )
        self.assertFalse(buckets[0]["partial"])
        self.assertFalse(buckets[1]["partial"])


class TestFormatFlat(unittest.TestCase):
    def _result(self):
        return {
            "days_back": 90,
            "window_start": "2026-01-16",
            "window_end": "2026-04-16",
            "signups": [
                {"client_id": 1, "contract_id": 10, "name": "Jane Smith",
                 "membership": "All Access 6 Month", "date": "2026-02-14"},
                {"client_id": 2, "contract_id": 11, "name": "Alex Ng",
                 "membership": "All Access 6 Month", "date": "2026-03-01"},
                {"client_id": 3, "contract_id": 12, "name": "Sam Lee",
                 "membership": "Student Membership", "date": "2026-02-20"},
            ],
            "cancellations": [
                {"client_id": 4, "contract_id": 13, "name": "John Doe",
                 "membership": "All Access 6 Month", "date": "2026-03-02"},
            ],
        }

    def test_header_and_counts(self):
        out = mindbody_helper.format_membership_movement(
            self._result(), days_back=90, split_by_month=False)
        self.assertIn("Membership Report", out)
        self.assertIn("Last 90 Days", out)
        self.assertIn("SIGNUPS: 3", out)
        self.assertIn("CANCELLATIONS: 1", out)
        self.assertIn("Net: +2", out)

    def test_groups_show_counts_not_names(self):
        out = mindbody_helper.format_membership_movement(
            self._result(), days_back=90, split_by_month=False)
        # Membership types are shown with counts
        self.assertIn("All Access 6 Month — 2", out)
        self.assertIn("Student Membership — 1", out)
        # Individual names are NOT rendered
        self.assertNotIn("Jane Smith", out)
        self.assertNotIn("Sam Lee", out)
        self.assertNotIn("John Doe", out)

    def test_empty_result(self):
        out = mindbody_helper.format_membership_movement(
            {"days_back": 30, "window_start": "2026-03-17", "window_end": "2026-04-16",
             "signups": [], "cancellations": []},
            days_back=30, split_by_month=False)
        self.assertIn("SIGNUPS: 0", out)
        self.assertIn("CANCELLATIONS: 0", out)
        self.assertIn("Net: 0", out)
        self.assertIn("(none)", out)


class TestFormatMonthly(unittest.TestCase):
    def _result(self):
        return {
            "days_back": 90,
            "window_start": "2026-01-16",
            "window_end": "2026-04-16",
            "signups": [
                {"client_id": 1, "contract_id": 10, "name": "Jane Smith",
                 "membership": "All Access 6 Month", "date": "2026-02-14"},
                {"client_id": 2, "contract_id": 11, "name": "Alex Ng",
                 "membership": "Student Membership", "date": "2026-03-01"},
            ],
            "cancellations": [
                {"client_id": 3, "contract_id": 12, "name": "John Doe",
                 "membership": "All Access 6 Month", "date": "2026-03-02"},
                {"client_id": 4, "contract_id": 13, "name": "Kim Lee",
                 "membership": "All Access 6 Month", "date": "2026-04-05"},
            ],
        }

    def test_header(self):
        out = mindbody_helper.format_membership_movement(
            self._result(), days_back=90, split_by_month=True)
        self.assertIn("Membership Report", out)
        self.assertIn("Last 90 Days", out)

    def test_each_month_section_present(self):
        out = mindbody_helper.format_membership_movement(
            self._result(), days_back=90, split_by_month=True)
        self.assertIn("January 2026", out)
        self.assertIn("February 2026", out)
        self.assertIn("March 2026", out)
        self.assertIn("April 2026", out)

    def test_partial_flag_rendered_on_edges(self):
        out = mindbody_helper.format_membership_movement(
            self._result(), days_back=90, split_by_month=True)
        self.assertIn("January 2026 (partial)", out)
        self.assertIn("April 2026 (partial)", out)
        self.assertNotIn("February 2026 (partial)", out)
        self.assertNotIn("March 2026 (partial)", out)

    def test_totals_footer(self):
        out = mindbody_helper.format_membership_movement(
            self._result(), days_back=90, split_by_month=True)
        self.assertIn("Signups: 2", out)
        self.assertIn("Cancellations: 2", out)
        self.assertIn("Net: 0", out)

    def test_counts_appear_under_correct_month(self):
        # Fixture: Feb has 1 All Access 6 Month signup, Mar has 1 Student
        # Membership signup + 1 All Access 6 Month cancellation, Apr has 1
        # All Access 6 Month cancellation.
        out = mindbody_helper.format_membership_movement(
            self._result(), days_back=90, split_by_month=True)
        feb_idx = out.find("February 2026")
        mar_idx = out.find("March 2026")
        apr_idx = out.find("April 2026")
        self.assertGreater(mar_idx, feb_idx)
        self.assertGreater(apr_idx, mar_idx)
        # Feb's signup line sits between the Feb header and Mar header
        feb_section = out[feb_idx:mar_idx]
        self.assertIn("All Access 6 Month — 1", feb_section)
        # Mar's section has the Student Membership signup and a cancellation
        mar_section = out[mar_idx:apr_idx]
        self.assertIn("Student Membership — 1", mar_section)
        self.assertIn("All Access 6 Month — 1", mar_section)
        # Apr's section has a cancellation
        apr_section = out[apr_idx:]
        self.assertIn("All Access 6 Month — 1", apr_section)

    def test_monthly_mode_via_mode_field(self):
        result = self._result()
        result["mode"] = "monthly"
        result["days_back"] = 90
        out = mindbody_helper.format_membership_movement(result)
        self.assertIn("January 2026", out)
        self.assertIn("Last 90 Days", out)


class TestTruncation(unittest.TestCase):
    def _big_result(self, n=300):
        # Each event uses a unique membership name to force many bullets,
        # overflowing the formatter cap even in counts-only mode.
        signups = [
            {"client_id": i, "contract_id": 1000 + i,
             "name": f"Client {i}",
             "membership": f"Test Membership Plan Number {i:03d}",
             "date": f"2026-02-{(i % 28) + 1:02d}"}
            for i in range(n)
        ]
        return {
            "days_back": 90,
            "window_start": "2026-01-16", "window_end": "2026-04-16",
            "signups": signups, "cancellations": [],
        }

    def test_flat_mode_respects_cap(self):
        out = mindbody_helper.format_membership_movement(
            self._big_result(), days_back=90, split_by_month=False)
        self.assertLessEqual(len(out), mindbody_helper.WHATSAPP_MAX_CHARS)

    def test_monthly_mode_respects_cap(self):
        out = mindbody_helper.format_membership_movement(
            self._big_result(), days_back=90, split_by_month=True)
        self.assertLessEqual(len(out), mindbody_helper.WHATSAPP_MAX_CHARS)

    def test_truncation_marker_present_when_truncated(self):
        out = mindbody_helper.format_membership_movement(
            self._big_result(), days_back=90, split_by_month=False)
        self.assertIn("truncated", out)


class TestFormatClientDetail(unittest.TestCase):
    def test_found_basic(self):
        result = {
            "status": "found",
            "name": "Jane Smith",
            "member_since": "2024-03-14",
            "memberships": ["All Access 6 Month"],
            "classes_all_time": 247,
            "classes_30d": 18,
            "classes_90d": 52,
        }
        out = mindbody_helper.format_client_detail(result)
        self.assertIn("Jane Smith", out)
        self.assertIn("14 March 2024", out)
        self.assertIn("All Access 6 Month", out)
        self.assertIn("247 (all time)", out)
        self.assertIn("Last 30 days: 18", out)
        self.assertIn("Last 90 days: 52", out)

    def test_found_with_custom_window(self):
        result = {
            "status": "found",
            "name": "Jane Smith",
            "member_since": "2024-03-14",
            "memberships": ["All Access 6 Month"],
            "classes_all_time": 247,
            "classes_30d": 18,
            "classes_90d": 52,
            "classes_custom": 6,
            "classes_custom_label": 14,
        }
        out = mindbody_helper.format_client_detail(result)
        self.assertIn("Last 14 days: 6", out)
        self.assertIn("Last 30 days: 18", out)
        self.assertIn("Last 90 days: 52", out)

    def test_found_multiple_memberships(self):
        result = {
            "status": "found",
            "name": "Jane Smith",
            "member_since": "2024-03-14",
            "memberships": ["All Access 6 Month", "Student Membership"],
            "classes_all_time": 10,
            "classes_30d": 3,
            "classes_90d": 8,
        }
        out = mindbody_helper.format_client_detail(result)
        self.assertIn("Current memberships:", out)
        self.assertIn("All Access 6 Month", out)
        self.assertIn("Student Membership", out)

    def test_found_no_active_membership(self):
        result = {
            "status": "found",
            "name": "Jane Smith",
            "member_since": "2024-03-14",
            "memberships": [],
            "classes_all_time": 50,
            "classes_30d": 0,
            "classes_90d": 0,
        }
        out = mindbody_helper.format_client_detail(result)
        self.assertIn("No active membership", out)

    def test_found_zero_visits(self):
        result = {
            "status": "found",
            "name": "Jane Smith",
            "member_since": "2024-03-14",
            "memberships": ["All Access 6 Month"],
            "classes_all_time": 0,
            "classes_30d": 0,
            "classes_90d": 0,
        }
        out = mindbody_helper.format_client_detail(result)
        self.assertIn("0 (all time)", out)
        self.assertIn("Last 30 days: 0", out)

    def test_not_found(self):
        result = {
            "status": "not_found",
            "search_text": "Janee Smyth",
        }
        out = mindbody_helper.format_client_detail(result)
        self.assertIn("No client found", out)
        self.assertIn("Janee Smyth", out)

    def test_multiple_matches(self):
        result = {
            "status": "multiple",
            "matches": [
                {"name": "Jane Smith", "id": 123, "email": "jane@email.com"},
                {"name": "Jane Doe", "id": 456, "email": "jdoe@email.com"},
            ],
        }
        out = mindbody_helper.format_client_detail(result)
        self.assertIn("Found 2 matches", out)
        self.assertIn("Jane Smith", out)
        self.assertIn("Jane Doe", out)
        self.assertIn("Which one", out)


class TestResolveWindow(unittest.TestCase):
    def test_neither_date_uses_days_back(self):
        mode, ws, we = mindbody_helper._resolve_movement_window(
            days_back=90, start_date=None, end_date=None,
            today_iso="2026-05-11",
        )
        self.assertEqual(mode, "monthly")
        self.assertEqual(we, "2026-05-11")
        self.assertEqual(ws, "2026-02-10")  # 90 days before 2026-05-11

    def test_both_dates_uses_range(self):
        mode, ws, we = mindbody_helper._resolve_movement_window(
            days_back=90, start_date="2026-03-01", end_date="2026-04-18",
            today_iso="2026-05-11",
        )
        self.assertEqual(mode, "range")
        self.assertEqual(ws, "2026-03-01")
        self.assertEqual(we, "2026-04-18")

    def test_only_start_date_defaults_end_to_today(self):
        mode, ws, we = mindbody_helper._resolve_movement_window(
            days_back=90, start_date="2026-03-01", end_date=None,
            today_iso="2026-05-11",
        )
        self.assertEqual(mode, "range")
        self.assertEqual(ws, "2026-03-01")
        self.assertEqual(we, "2026-05-11")

    def test_only_end_date_defaults_start_to_end_minus_90(self):
        mode, ws, we = mindbody_helper._resolve_movement_window(
            days_back=90, start_date=None, end_date="2026-04-18",
            today_iso="2026-05-11",
        )
        self.assertEqual(mode, "range")
        self.assertEqual(we, "2026-04-18")
        self.assertEqual(ws, "2026-01-18")  # 90 days before 2026-04-18

    def test_invalid_start_date_format_raises(self):
        with self.assertRaises(ValueError) as ctx:
            mindbody_helper._resolve_movement_window(
                days_back=90, start_date="03/01/2026", end_date=None,
                today_iso="2026-05-11",
            )
        self.assertIn("YYYY-MM-DD", str(ctx.exception))

    def test_invalid_end_date_format_raises(self):
        with self.assertRaises(ValueError) as ctx:
            mindbody_helper._resolve_movement_window(
                days_back=90, start_date=None, end_date="not-a-date",
                today_iso="2026-05-11",
            )
        self.assertIn("YYYY-MM-DD", str(ctx.exception))

    def test_start_after_end_raises(self):
        with self.assertRaises(ValueError) as ctx:
            mindbody_helper._resolve_movement_window(
                days_back=90, start_date="2026-04-18", end_date="2026-03-01",
                today_iso="2026-05-11",
            )
        self.assertIn("on or before", str(ctx.exception))

    def test_range_over_365_days_raises(self):
        with self.assertRaises(ValueError) as ctx:
            mindbody_helper._resolve_movement_window(
                days_back=90, start_date="2024-01-01", end_date="2026-04-18",
                today_iso="2026-05-11",
            )
        self.assertIn("365", str(ctx.exception))

    def test_exactly_365_days_allowed(self):
        mode, ws, we = mindbody_helper._resolve_movement_window(
            days_back=90,
            start_date="2025-01-01", end_date="2026-01-01",  # 365-day span
            today_iso="2026-05-11",
        )
        self.assertEqual(mode, "range")
        self.assertEqual(ws, "2025-01-01")
        self.assertEqual(we, "2026-01-01")

    def test_only_end_date_default_start_ignores_days_back(self):
        # Confirms the start-side default is always end_date - 90 days,
        # not end_date - days_back days, when only end_date is provided.
        mode, ws, we = mindbody_helper._resolve_movement_window(
            days_back=30,
            start_date=None, end_date="2026-04-18",
            today_iso="2026-05-11",
        )
        self.assertEqual(mode, "range")
        self.assertEqual(we, "2026-04-18")
        self.assertEqual(ws, "2026-01-18")  # always 90 days before end_date

    def test_same_start_and_end_allowed(self):
        mode, ws, we = mindbody_helper._resolve_movement_window(
            days_back=90, start_date="2026-04-18", end_date="2026-04-18",
            today_iso="2026-05-11",
        )
        self.assertEqual(mode, "range")
        self.assertEqual(ws, "2026-04-18")
        self.assertEqual(we, "2026-04-18")


class TestRangeHeader(unittest.TestCase):
    def test_same_year_no_year_on_left(self):
        out = mindbody_helper._format_range_header("2026-03-01", "2026-04-18")
        self.assertEqual(out, "Mar 1 — Apr 18, 2026")

    def test_same_month_same_year(self):
        out = mindbody_helper._format_range_header("2026-04-01", "2026-04-18")
        self.assertEqual(out, "Apr 1 — Apr 18, 2026")

    def test_same_day(self):
        out = mindbody_helper._format_range_header("2026-04-18", "2026-04-18")
        self.assertEqual(out, "Apr 18 — Apr 18, 2026")

    def test_cross_year_includes_both_years(self):
        out = mindbody_helper._format_range_header("2025-12-15", "2026-01-14")
        self.assertEqual(out, "Dec 15, 2025 — Jan 14, 2026")


class TestFormatRange(unittest.TestCase):
    def _result(self):
        return {
            "mode": "range",
            "window_start": "2026-03-01",
            "window_end": "2026-04-18",
            "signups": [
                {"client_id": 1, "contract_id": 10, "name": "Jane Smith",
                 "membership": "All Access 6 Month", "date": "2026-03-14"},
                {"client_id": 2, "contract_id": 11, "name": "Alex Ng",
                 "membership": "All Access 6 Month", "date": "2026-04-01"},
                {"client_id": 3, "contract_id": 12, "name": "Sam Lee",
                 "membership": "Student Membership", "date": "2026-04-10"},
            ],
            "cancellations": [
                {"client_id": 4, "contract_id": 13, "name": "John Doe",
                 "membership": "All Access 6 Month", "date": "2026-03-22"},
            ],
        }

    def test_header_uses_range_dates_not_days_back(self):
        out = mindbody_helper.format_membership_movement(self._result())
        self.assertIn("Membership Report", out)
        self.assertIn("Mar 1 — Apr 18, 2026", out)
        self.assertNotIn("Last", out)  # no "Last N Days" header in range mode

    def test_counts_and_net(self):
        out = mindbody_helper.format_membership_movement(self._result())
        self.assertIn("SIGNUPS: 3", out)
        self.assertIn("CANCELLATIONS: 1", out)
        self.assertIn("Net: +2", out)

    def test_counts_by_membership_no_names(self):
        out = mindbody_helper.format_membership_movement(self._result())
        self.assertIn("All Access 6 Month — 2", out)
        self.assertIn("Student Membership — 1", out)
        self.assertNotIn("Jane Smith", out)
        self.assertNotIn("Sam Lee", out)
        self.assertNotIn("John Doe", out)

    def test_empty_range(self):
        out = mindbody_helper.format_membership_movement({
            "mode": "range",
            "window_start": "2026-04-01",
            "window_end": "2026-04-18",
            "signups": [], "cancellations": [],
        })
        self.assertIn("Apr 1 — Apr 18, 2026", out)
        self.assertIn("SIGNUPS: 0", out)
        self.assertIn("CANCELLATIONS: 0", out)
        self.assertIn("Net: 0", out)
        self.assertIn("(none)", out)

    def test_negative_net(self):
        result = self._result()
        result["signups"] = []
        out = mindbody_helper.format_membership_movement(result)
        self.assertIn("SIGNUPS: 0", out)
        self.assertIn("CANCELLATIONS: 1", out)
        self.assertIn("Net: -1", out)

    def test_respects_whatsapp_cap(self):
        result = {
            "mode": "range",
            "window_start": "2026-01-01",
            "window_end": "2026-04-18",
            "signups": [
                {"client_id": i, "contract_id": 1000 + i,
                 "name": f"Client {i}",
                 "membership": f"Test Plan {i:03d}",
                 "date": "2026-02-14"}
                for i in range(300)
            ],
            "cancellations": [],
        }
        out = mindbody_helper.format_membership_movement(result)
        self.assertLessEqual(len(out), mindbody_helper.WHATSAPP_MAX_CHARS)


class TestGetMembershipMovementInputs(unittest.TestCase):
    """Behavioural tests that mock the MindBody fetch so we exercise the
    window-resolution + result-shape changes without hitting the API."""

    def setUp(self):
        # Reset any cached result between tests
        mindbody_helper._cache.clear()

    def _patch_fetch(self, candidates=None, contracts_by_client=None):
        candidates = candidates or []
        contracts_by_client = contracts_by_client or {}

        def fake_paginated(*args, **kwargs):
            return candidates

        def fake_api_get(path, params=None):
            if path == "client/clientcontracts":
                cid = params.get("ClientId")
                return {"Contracts": contracts_by_client.get(cid, [])}
            return {}

        return fake_paginated, fake_api_get

    def test_default_call_returns_monthly_mode(self):
        fake_paginated, fake_api_get = self._patch_fetch()
        orig_p = mindbody_helper._get_all_clients_paginated
        orig_g = mindbody_helper._api_get
        mindbody_helper._get_all_clients_paginated = fake_paginated
        mindbody_helper._api_get = fake_api_get
        try:
            result = mindbody_helper.get_membership_movement()
        finally:
            mindbody_helper._get_all_clients_paginated = orig_p
            mindbody_helper._api_get = orig_g
        self.assertEqual(result["mode"], "monthly")
        self.assertEqual(result["days_back"], 90)

    def test_explicit_dates_return_range_mode(self):
        fake_paginated, fake_api_get = self._patch_fetch()
        orig_p = mindbody_helper._get_all_clients_paginated
        orig_g = mindbody_helper._api_get
        mindbody_helper._get_all_clients_paginated = fake_paginated
        mindbody_helper._api_get = fake_api_get
        try:
            result = mindbody_helper.get_membership_movement(
                start_date="2026-03-01", end_date="2026-04-18",
            )
        finally:
            mindbody_helper._get_all_clients_paginated = orig_p
            mindbody_helper._api_get = orig_g
        self.assertEqual(result["mode"], "range")
        self.assertEqual(result["window_start"], "2026-03-01")
        self.assertEqual(result["window_end"], "2026-04-18")
        self.assertIsNone(result["days_back"])

    def test_invalid_date_raises_valueerror(self):
        with self.assertRaises(ValueError):
            mindbody_helper.get_membership_movement(start_date="bogus")


if __name__ == "__main__":
    unittest.main()
