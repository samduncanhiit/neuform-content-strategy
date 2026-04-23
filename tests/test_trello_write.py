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


class TestFindCards(unittest.TestCase):
    @patch("trello_helper._trello_get")
    def test_returns_sorted_matches(self, mock_get):
        mock_get.side_effect = [
            [
                {"id": "c1", "name": "Buy kettlebells", "idList": "l1",
                 "shortUrl": "https://trello.com/c/c1"},
                {"id": "c2", "name": "Order dumbbells", "idList": "l1",
                 "shortUrl": "https://trello.com/c/c2"},
                {"id": "c3", "name": "kettlebell rack install", "idList": "l2",
                 "shortUrl": "https://trello.com/c/c3"},
            ],
            [
                {"id": "l1", "name": "To Do List"},
                {"id": "l2", "name": "Doing"},
            ],
        ]
        matches = trello_helper._find_cards("board1", "kettlebell")
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0]["name"], "Buy kettlebells")
        self.assertEqual(matches[0]["list_name"], "To Do List")
        self.assertEqual(matches[0]["url"], "https://trello.com/c/c1")

    @patch("trello_helper._trello_get")
    def test_no_matches_returns_empty(self, mock_get):
        mock_get.side_effect = [
            [{"id": "c1", "name": "Pay power bill", "idList": "l1", "shortUrl": ""}],
            [{"id": "l1", "name": "Bills"}],
        ]
        self.assertEqual(trello_helper._find_cards("board1", "kettlebell"), [])


class TestResolveLabels(unittest.TestCase):
    @patch("trello_helper._trello_post")
    @patch("trello_helper._trello_get")
    def test_returns_existing_label_ids_and_creates_missing(self, mock_get, mock_post):
        mock_get.return_value = [
            {"id": "lab1", "name": "Urgent"},
            {"id": "lab2", "name": "Admin"},
        ]
        mock_post.return_value = {"id": "lab3", "name": "New"}

        ids = trello_helper._resolve_labels("board1", ["Urgent", "New"])

        self.assertEqual(ids, ["lab1", "lab3"])
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "boards/board1/labels")
        self.assertEqual(kwargs.get("params", {}).get("name"), "New")

    @patch("trello_helper._trello_get")
    def test_empty_list_returns_empty(self, mock_get):
        self.assertEqual(trello_helper._resolve_labels("board1", []), [])
        mock_get.assert_not_called()

    @patch("trello_helper._trello_post")
    @patch("trello_helper._trello_get")
    def test_case_insensitive_match(self, mock_get, mock_post):
        mock_get.return_value = [{"id": "lab1", "name": "Urgent"}]
        ids = trello_helper._resolve_labels("board1", ["urgent"])
        self.assertEqual(ids, ["lab1"])
        mock_post.assert_not_called()


class TestCardWriteWrappers(unittest.TestCase):
    @patch("trello_helper._trello_post")
    def test_create_card_sends_required_fields(self, mock_post):
        mock_post.return_value = {"id": "c1", "shortUrl": "https://trello.com/c/c1"}
        result = trello_helper.create_card(
            board_id="b1", list_id="l1", title="Buy kettlebells",
            due_date="2026-04-25", description="From supplier X",
            label_ids=["lab1", "lab2"],
        )
        self.assertEqual(result["id"], "c1")
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "cards")
        params = kwargs.get("params", {})
        self.assertEqual(params["idList"], "l1")
        self.assertEqual(params["name"], "Buy kettlebells")
        self.assertEqual(params["due"], "2026-04-25T10:00:00.000Z")
        self.assertEqual(params["desc"], "From supplier X")
        self.assertEqual(params["idLabels"], "lab1,lab2")

    @patch("trello_helper._trello_post")
    def test_create_card_without_optional_fields(self, mock_post):
        mock_post.return_value = {"id": "c1", "shortUrl": ""}
        trello_helper.create_card(board_id="b1", list_id="l1", title="t")
        args, kwargs = mock_post.call_args
        params = kwargs.get("params", {})
        self.assertEqual(params["name"], "t")
        self.assertNotIn("due", params)
        self.assertNotIn("desc", params)
        self.assertNotIn("idLabels", params)

    @patch("trello_helper._trello_put")
    def test_update_card_passes_through_fields(self, mock_put):
        mock_put.return_value = {"id": "c1"}
        trello_helper.update_card(
            "c1", name="New title", due_date="2026-05-01",
            description="New desc", label_ids=["lab1"],
        )
        args, kwargs = mock_put.call_args
        self.assertEqual(args[0], "cards/c1")
        params = kwargs.get("params", {})
        self.assertEqual(params["name"], "New title")
        self.assertEqual(params["due"], "2026-05-01T10:00:00.000Z")
        self.assertEqual(params["desc"], "New desc")
        self.assertEqual(params["idLabels"], "lab1")

    @patch("trello_helper._trello_put")
    def test_update_card_skips_none_fields(self, mock_put):
        mock_put.return_value = {"id": "c1"}
        trello_helper.update_card("c1", name="Only title")
        args, kwargs = mock_put.call_args
        params = kwargs.get("params", {})
        self.assertEqual(list(params.keys()), ["name"])

    @patch("trello_helper._trello_put")
    def test_move_card_sends_idList(self, mock_put):
        mock_put.return_value = {"id": "c1"}
        trello_helper.move_card("c1", "l2")
        args, kwargs = mock_put.call_args
        self.assertEqual(args[0], "cards/c1")
        self.assertEqual(kwargs.get("params", {}), {"idList": "l2"})

    @patch("trello_helper._trello_put")
    def test_archive_card_sets_closed_true(self, mock_put):
        mock_put.return_value = {"id": "c1"}
        trello_helper.archive_card("c1")
        args, kwargs = mock_put.call_args
        self.assertEqual(args[0], "cards/c1")
        self.assertEqual(kwargs.get("params", {}), {"closed": "true"})


class TestAddTrelloCardHandler(unittest.TestCase):
    def setUp(self):
        import app
        self.app = app

    @patch("trello_helper.create_card")
    @patch("trello_helper._resolve_labels")
    @patch("trello_helper._find_list")
    @patch("trello_helper._find_board_id")
    def test_adds_card_with_defaults(self, mock_board, mock_list, mock_labels, mock_create):
        mock_board.return_value = "b1"
        mock_list.return_value = ("l1", "To Do List")
        mock_labels.return_value = []
        mock_create.return_value = {"id": "c1", "shortUrl": "https://trello.com/c/c1"}

        result = self.app.handle_tool_call(
            "add_trello_card",
            {"title": "Buy kettlebells"},
            raw_number="+61421188443",
        )
        self.assertIn("Buy kettlebells", result)
        self.assertIn("To Do List", result)
        self.assertIn("HIIT Office", result)
        mock_list.assert_called_with("b1", "To Do List")
        mock_create.assert_called_once()

    @patch("trello_helper._find_board_id")
    def test_refuses_unconfigured_user(self, mock_board):
        result = self.app.handle_tool_call(
            "add_trello_card",
            {"title": "X"},
            raw_number="+61400000000",
        )
        self.assertIn("not configured", result.lower())
        mock_board.assert_not_called()

    @patch("trello_helper._find_list")
    @patch("trello_helper._find_board_id")
    def test_uses_list_name_override(self, mock_board, mock_list):
        mock_board.return_value = "b1"
        mock_list.return_value = ("l9", "Revenue Growth Ideas")
        with patch("trello_helper.create_card") as mock_create, \
             patch("trello_helper._resolve_labels", return_value=[]):
            mock_create.return_value = {"id": "c1", "shortUrl": ""}
            self.app.handle_tool_call(
                "add_trello_card",
                {"title": "X", "list_name": "Revenue Growth Ideas"},
                raw_number="+61421188443",
            )
        mock_list.assert_called_with("b1", "Revenue Growth Ideas")
