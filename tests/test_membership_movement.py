"""Unit tests for the membership movement report pure functions."""
import unittest

import mindbody_helper


class TestScaffolding(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(hasattr(mindbody_helper, "_is_tracked_membership"))


if __name__ == "__main__":
    unittest.main()
