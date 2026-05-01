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


class TestApiGet(unittest.TestCase):
    @patch("jotform_helper.JOTFORM_API_KEY", "fake-key")
    @patch("jotform_helper.requests.get")
    def test_calls_correct_url_with_api_key(self, mock_get):
        mock_resp = mock_get.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"content": []}
        jotform_helper._api_get("/user/forms", {"limit": 1000})
        args, kwargs = mock_get.call_args
        self.assertEqual(args[0], "https://api.jotform.com/user/forms")
        self.assertEqual(kwargs["params"]["apiKey"], "fake-key")
        self.assertEqual(kwargs["params"]["limit"], 1000)

    @patch("jotform_helper.JOTFORM_API_KEY", None)
    def test_raises_when_api_key_missing(self):
        with self.assertRaises(RuntimeError) as cm:
            jotform_helper._api_get("/user/forms")
        self.assertIn("JOTFORM_API_KEY", str(cm.exception))

    @patch("jotform_helper.JOTFORM_API_KEY", "fake-key")
    @patch("jotform_helper.requests.get")
    def test_returns_content_field_from_response(self, mock_get):
        mock_resp = mock_get.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "responseCode": 200,
            "content": [{"id": "f1", "title": "A"}],
        }
        result = jotform_helper._api_get("/user/forms")
        self.assertEqual(result, [{"id": "f1", "title": "A"}])
