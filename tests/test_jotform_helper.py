"""Unit tests for jotform_helper."""
import unittest
from unittest.mock import patch

import jotform_helper


class TestScaffolding(unittest.TestCase):
    def test_module_exposes_public_helpers(self):
        for name in [
            "_api_get",
            "_find_forms_by_name",
            "get_submission_count",
        ]:
            self.assertTrue(
                hasattr(jotform_helper, name),
                f"missing {name}",
            )
