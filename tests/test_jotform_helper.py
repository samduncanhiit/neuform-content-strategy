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

    @patch("jotform_helper.JOTFORM_API_KEY", "fake-key")
    @patch("jotform_helper.requests.get")
    def test_raises_on_http_error(self, mock_get):
        from requests import HTTPError
        mock_get.return_value.raise_for_status.side_effect = HTTPError("401")
        with self.assertRaises(HTTPError):
            jotform_helper._api_get("/user/forms")


def _form(form_id, title, count, status="ENABLED"):
    return {"id": form_id, "title": title, "count": str(count), "status": status}


class TestFindFormsByName(unittest.TestCase):
    @patch("jotform_helper._api_get")
    def test_exact_match_case_insensitive(self, mock_get):
        mock_get.return_value = [
            _form("1", "Lead Form", 12),
            _form("2", "Other Form", 5),
        ]
        result = jotform_helper._find_forms_by_name("lead form")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "1")

    @patch("jotform_helper._api_get")
    def test_substring_match_when_no_exact(self, mock_get):
        mock_get.return_value = [
            _form("1", "Lead Form V1", 12),
            _form("2", "Lead Form V2", 7),
            _form("3", "Other Form", 5),
        ]
        result = jotform_helper._find_forms_by_name("lead form")
        self.assertEqual({f["id"] for f in result}, {"1", "2"})

    @patch("jotform_helper._api_get")
    def test_no_match_returns_empty(self, mock_get):
        mock_get.return_value = [_form("1", "Other Form", 5)]
        self.assertEqual(jotform_helper._find_forms_by_name("nope"), [])

    @patch("jotform_helper._api_get")
    def test_filters_out_deleted_forms(self, mock_get):
        mock_get.return_value = [
            _form("1", "Lead Form", 12, status="DELETED"),
            _form("2", "Lead Form", 7, status="ENABLED"),
        ]
        result = jotform_helper._find_forms_by_name("lead form")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "2")

    @patch("jotform_helper._api_get")
    def test_exact_match_takes_priority_over_substring(self, mock_get):
        mock_get.return_value = [
            _form("1", "Lead Form", 12),
            _form("2", "Lead Form V2", 7),
        ]
        result = jotform_helper._find_forms_by_name("lead form")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "1")
