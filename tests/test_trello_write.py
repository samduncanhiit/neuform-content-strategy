"""Unit tests for Trello write helpers and tool handlers."""
import unittest
from unittest.mock import patch, MagicMock

import trello_helper


class TestScaffolding(unittest.TestCase):
    def test_module_exposes_write_helpers(self):
        for name in [
            "_trello_post", "_trello_put",
            "_find_board_id", "_find_list", "_find_cards",
            "_fuzzy_score", "_resolve_labels",
            "create_card", "update_card", "move_card", "archive_card",
        ]:
            self.assertTrue(hasattr(trello_helper, name), f"missing {name}")


class TestFuzzyScore(unittest.TestCase):
    def test_exact_match_is_highest(self):
        self.assertGreater(
            trello_helper._fuzzy_score("buy kettlebells", "buy kettlebells"),
            trello_helper._fuzzy_score("buy kettlebells", "order dumbbells"),
        )

    def test_substring_match_beats_non_match(self):
        self.assertGreater(
            trello_helper._fuzzy_score("kettlebell", "buy kettlebells for gym"),
            trello_helper._fuzzy_score("kettlebell", "order new mats"),
        )

    def test_case_insensitive(self):
        self.assertEqual(
            trello_helper._fuzzy_score("Kettlebell", "buy kettlebell"),
            trello_helper._fuzzy_score("kettlebell", "BUY KETTLEBELL"),
        )

    def test_token_overlap_scores(self):
        self.assertGreater(
            trello_helper._fuzzy_score("kettlebell order", "order 4 kettlebells"),
            trello_helper._fuzzy_score("kettlebell order", "pay power bill"),
        )

    def test_empty_needle_returns_zero(self):
        self.assertEqual(trello_helper._fuzzy_score("", "anything"), 0)

    def test_no_overlap_returns_zero(self):
        self.assertEqual(
            trello_helper._fuzzy_score("kettlebell", "pay power bill"),
            0,
        )
