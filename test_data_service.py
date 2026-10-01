import copy
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile, score_prospect
from gtm_agent.gtm_records import OFFERINGS, PROSPECTS


class ProspectTechnologyUpdateTest(unittest.TestCase):
    prospect_id = "LEAD-39002"

    def setUp(self):
        self.original_record = copy.deepcopy(PROSPECTS[self.prospect_id])
        PROSPECTS[self.prospect_id] = copy.deepcopy(self.original_record)
        data_service._PROFILES.clear()

    def tearDown(self):
        PROSPECTS[self.prospect_id] = self.original_record
        data_service._PROFILES.clear()

    def test_update_persists_refreshes_profile_and_scoring_uses_saved_stack(self):
        initial_profile = build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertNotIn("Terraform", initial_profile["prospect_profile"]["tech_stack"])

        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertTrue(result["updated"])
        self.assertIn("Terraform", data_service.fetch_tech_stack(self.prospect_id))
        refreshed_profile = build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertIn("Terraform", refreshed_profile["prospect_profile"]["tech_stack"])

        scoring_result = Mock(model_dump=lambda: {"justification": "Terraform is present."})
        with patch("gtm_agent.gtm_agent._scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value = scoring_result
            score_prospect.invoke({
                "prospect_profile": initial_profile["prospect_profile"],
                "offering": OFFERINGS["OFFER-10004"],
            })

        scored_profile = scoring_llm.invoke.call_args.args[0][1]["content"]
        self.assertIn('"Terraform"', scored_profile)


if __name__ == "__main__":
    unittest.main()
