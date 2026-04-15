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


class TestTruncation(unittest.TestCase):
    def _big_result(self, n=100):
        # Each event uses a unique membership name to force many bullets,
        # overflowing the WhatsApp cap even in counts-only mode.
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


if __name__ == "__main__":
    unittest.main()
