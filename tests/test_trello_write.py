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
