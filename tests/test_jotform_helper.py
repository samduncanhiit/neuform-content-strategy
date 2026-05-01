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


class TestGetSubmissionCount(unittest.TestCase):
    @patch("jotform_helper._find_forms_by_name")
    def test_single_match_returns_ok(self, mock_find):
        mock_find.return_value = [_form("1", "Lead Form", 42)]
        result = jotform_helper.get_submission_count("lead form")
        self.assertEqual(result, {
            "status": "ok",
            "title": "Lead Form",
            "count": 42,
        })

    @patch("jotform_helper._suggest_form")
    @patch("jotform_helper._find_forms_by_name")
    def test_no_match_returns_none_status(self, mock_find, mock_suggest):
        mock_find.return_value = []
        mock_suggest.return_value = None
        result = jotform_helper.get_submission_count("nope")
        self.assertEqual(result, {"status": "none", "name": "nope"})

    @patch("jotform_helper._find_forms_by_name")
    def test_multiple_matches_returns_multiple(self, mock_find):
        mock_find.return_value = [
            _form("1", "Lead Form V1", 12),
            _form("2", "Lead Form V2", 7),
        ]
        result = jotform_helper.get_submission_count("lead form")
        self.assertEqual(result["status"], "multiple")
        self.assertEqual(result["matches"], [
            {"title": "Lead Form V1", "count": 12},
            {"title": "Lead Form V2", "count": 7},
        ])

    @patch("jotform_helper._find_forms_by_name")
    def test_missing_api_key_returns_error(self, mock_find):
        mock_find.side_effect = RuntimeError("JOTFORM_API_KEY env var is not set")
        result = jotform_helper.get_submission_count("anything")
        self.assertEqual(result["status"], "error")
        self.assertIn("not configured", result["message"].lower())

    @patch("jotform_helper._find_forms_by_name")
    def test_http_401_returns_error(self, mock_find):
        from requests import HTTPError, Response
        resp = Response()
        resp.status_code = 401
        mock_find.side_effect = HTTPError(response=resp)
        result = jotform_helper.get_submission_count("anything")
        self.assertEqual(result["status"], "error")
        self.assertIn("invalid", result["message"].lower())

    @patch("jotform_helper._find_forms_by_name")
    def test_network_error_returns_error(self, mock_find):
        from requests import ConnectionError as ReqConnErr
        mock_find.side_effect = ReqConnErr("boom")
        result = jotform_helper.get_submission_count("anything")
        self.assertEqual(result["status"], "error")
        self.assertIn("try again", result["message"].lower())

    @patch("jotform_helper._find_forms_by_name")
    def test_count_is_coerced_to_int(self, mock_find):
        # JotForm returns count as a string — make sure we coerce.
        mock_find.return_value = [_form("1", "Lead Form", "99")]
        result = jotform_helper.get_submission_count("lead form")
        self.assertEqual(result["count"], 99)

    @patch("jotform_helper._suggest_form")
    @patch("jotform_helper._find_forms_by_name")
    def test_no_match_with_suggestion_returns_suggest(self, mock_find, mock_suggest):
        mock_find.return_value = []
        mock_suggest.return_value = _form("1", "Labour day Lower body Strength session", 40)
        result = jotform_helper.get_submission_count("lower body")
        self.assertEqual(result, {
            "status": "suggest",
            "name": "lower body",
            "suggestion": {
                "title": "Labour day Lower body Strength session",
                "count": 40,
            },
        })

    @patch("jotform_helper._suggest_form")
    @patch("jotform_helper._find_forms_by_name")
    def test_no_match_no_suggestion_returns_none(self, mock_find, mock_suggest):
        mock_find.return_value = []
        mock_suggest.return_value = None
        result = jotform_helper.get_submission_count("kettlebell")
        self.assertEqual(result, {"status": "none", "name": "kettlebell"})

    @patch("jotform_helper._suggest_form")
    @patch("jotform_helper._find_forms_by_name")
    def test_suggestion_count_is_coerced_to_int(self, mock_find, mock_suggest):
        mock_find.return_value = []
        # JotForm returns count as a string; suggestion count should also be coerced.
        mock_suggest.return_value = _form("1", "Lead Form", "99")
        result = jotform_helper.get_submission_count("lead")
        self.assertEqual(result["suggestion"]["count"], 99)


class TestFuzzyScore(unittest.TestCase):
    def test_exact_match_is_highest(self):
        self.assertGreater(
            jotform_helper._fuzzy_score("lower body", "lower body"),
            jotform_helper._fuzzy_score("lower body", "upper body"),
        )

    def test_substring_match_beats_token_only(self):
        # substring scores 10+, token-only scores 1-9
        self.assertGreaterEqual(
            jotform_helper._fuzzy_score("strength", "lower body strength"),
            10,
        )

    def test_case_insensitive(self):
        self.assertEqual(
            jotform_helper._fuzzy_score("Strength", "lower body strength"),
            jotform_helper._fuzzy_score("strength", "LOWER BODY STRENGTH"),
        )

    def test_token_overlap_scores(self):
        # "challenge round" should match "8 Week Challenge Round 14" via tokens
        score = jotform_helper._fuzzy_score(
            "challenge round", "8 Week Challenge Round 14"
        )
        self.assertGreater(score, 0)

    def test_empty_needle_returns_zero(self):
        self.assertEqual(jotform_helper._fuzzy_score("", "anything"), 0)

    def test_empty_haystack_returns_zero(self):
        self.assertEqual(jotform_helper._fuzzy_score("anything", ""), 0)

    def test_no_overlap_returns_zero(self):
        self.assertEqual(
            jotform_helper._fuzzy_score("kettlebell", "labour day breakfast"),
            0,
        )


class TestSuggestForm(unittest.TestCase):
    @patch("jotform_helper._api_get")
    def test_returns_highest_scoring_form(self, mock_get):
        mock_get.return_value = [
            _form("1", "Lower Body Strength Session", 40),
            _form("2", "Upper Body Hypertrophy", 12),
            _form("3", "Cardio Bootcamp", 5),
        ]
        # "lower body strength" is a substring of form 1's title — score 10+
        result = jotform_helper._suggest_form("lower body strength")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "1")

    @patch("jotform_helper._api_get")
    def test_returns_token_match_when_no_substring(self, mock_get):
        mock_get.return_value = [
            _form("1", "8 Week Challenge Round 14", 36),
            _form("2", "Cardio Bootcamp", 5),
        ]
        # "challenge week" isn't a substring of either title (form 1 has "Week Challenge"
        # in that order), but tokens "challenge" and "week" both match form 1 → token-only
        # score = 2. Form 2 shares no tokens → score = 0. Form 1 wins.
        result = jotform_helper._suggest_form("challenge week")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "1")

    @patch("jotform_helper._api_get")
    def test_returns_none_when_no_form_scores_above_threshold(self, mock_get):
        mock_get.return_value = [
            _form("1", "Cardio Bootcamp", 5),
            _form("2", "End of Challenge Party", 0),
        ]
        # "kettlebell" shares no tokens with either title → all score 0
        result = jotform_helper._suggest_form("kettlebell")
        self.assertIsNone(result)

    @patch("jotform_helper._api_get")
    def test_filters_out_deleted_forms(self, mock_get):
        mock_get.return_value = [
            _form("1", "Lower Body Strength", 40, status="DELETED"),
            _form("2", "Cardio Bootcamp", 5),
        ]
        # The deleted form would be the closest match for "lower body" but should be skipped.
        result = jotform_helper._suggest_form("lower body")
        # Cardio Bootcamp scores 0 against "lower body", so we expect None.
        self.assertIsNone(result)

    @patch("jotform_helper._api_get")
    def test_empty_name_returns_none(self, mock_get):
        # No API call needed for empty input.
        result = jotform_helper._suggest_form("")
        self.assertIsNone(result)
        mock_get.assert_not_called()

    @patch("jotform_helper._api_get")
    def test_whitespace_only_name_returns_none(self, mock_get):
        result = jotform_helper._suggest_form("   ")
        self.assertIsNone(result)
        mock_get.assert_not_called()
