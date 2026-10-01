import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def call_tool(self, prospect):
        return send_prospect_email.func(
            prospect,
            "Book a demo",
            "Please book a demo.",
            None,
            {"name": "Ava Mills", "email": "ava@example.com"},
        )

    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record")
    def test_disqualified_source_record_blocks_send(self, get_prospect_record):
        get_prospect_record.return_value = {
            "name": "Source Name",
            "email": "source@example.com",
            "disqualified": True,
        }

        result = self.call_tool({
            "prospect_id": "LEAD-50001",
            "name": "Model Name",
            "email": "model@example.com",
            "disqualified": False,
        })

        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
        })

    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record")
    def test_missing_source_record_blocks_send(self, get_prospect_record):
        get_prospect_record.return_value = None

        result = self.call_tool({"prospect_id": "LEAD-UNKNOWN", "email": "model@example.com"})

        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect record not found; email not sent.",
        })

    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record")
    def test_qualified_source_record_allows_send(self, get_prospect_record):
        get_prospect_record.return_value = {
            "name": "Qualified Name",
            "email": "qualified@example.com",
            "disqualified": False,
        }

        result = self.call_tool({
            "prospect_id": "LEAD-60001",
            "name": "Model Name",
            "email": "model@example.com",
        })

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "qualified@example.com")
        self.assertEqual(result["to_name"], "Qualified Name")

    def test_prompt_requires_disqualified_prospects_to_be_blocked(self):
        self.assertIn("Prospects marked disqualified must not be emailed", SYSTEM_PROMPT)
        self.assertIn("tell the rep the prospect is disqualified", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
