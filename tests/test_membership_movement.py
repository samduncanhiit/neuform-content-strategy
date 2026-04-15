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


if __name__ == "__main__":
    unittest.main()
