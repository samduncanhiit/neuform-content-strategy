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


class TestFindBoard(unittest.TestCase):
    def setUp(self):
        trello_helper._board_id_cache_by_name = {}

    @patch("trello_helper._trello_get")
    def test_finds_board_by_exact_name(self, mock_get):
        mock_get.return_value = [
            {"id": "b1", "name": "HIIT Office"},
            {"id": "b2", "name": "HIIT Challenge"},
        ]
        self.assertEqual(trello_helper._find_board_id("HIIT Office"), "b1")

    @patch("trello_helper._trello_get")
    def test_is_case_insensitive(self, mock_get):
        mock_get.return_value = [{"id": "b1", "name": "HIIT Office"}]
        self.assertEqual(trello_helper._find_board_id("hiit office"), "b1")

    @patch("trello_helper._trello_get")
    def test_returns_none_when_no_match(self, mock_get):
        mock_get.return_value = [{"id": "b1", "name": "Other Board"}]
        self.assertIsNone(trello_helper._find_board_id("HIIT Office"))

    @patch("trello_helper._trello_get")
    def test_caches_result(self, mock_get):
        mock_get.return_value = [{"id": "b1", "name": "HIIT Office"}]
        trello_helper._find_board_id("HIIT Office")
        trello_helper._find_board_id("HIIT Office")
        self.assertEqual(mock_get.call_count, 1)


class TestFindList(unittest.TestCase):
    def setUp(self):
        trello_helper._list_cache_by_board = {}

    @patch("trello_helper._trello_get")
    def test_finds_list_by_exact_name(self, mock_get):
        mock_get.return_value = [
            {"id": "l1", "name": "To Do List"},
            {"id": "l2", "name": "Done"},
        ]
        result = trello_helper._find_list("board1", "Done")
        self.assertEqual(result, ("l2", "Done"))

    @patch("trello_helper._trello_get")
    def test_is_case_insensitive(self, mock_get):
        mock_get.return_value = [{"id": "l1", "name": "To Do List"}]
        result = trello_helper._find_list("board1", "to do list")
        self.assertEqual(result, ("l1", "To Do List"))

    @patch("trello_helper._trello_get")
    def test_returns_none_when_no_match(self, mock_get):
        mock_get.return_value = [{"id": "l1", "name": "Done"}]
        self.assertIsNone(trello_helper._find_list("board1", "Nonexistent"))
