import json
import os
import re
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service, gtm_agent
from gtm_agent.gtm_records import OFFERINGS, PROSPECTS


class _ScoringResult:
    def model_dump(self):
        return {"score": 50}


class _ScoringLLM:
    def __init__(self):
        self.payloads = []

    def invoke(self, messages):
        self.payloads.append(messages)
        return _ScoringResult()


class ProspectPIIBoundaryTests(unittest.TestCase):
    def test_tools_and_scoring_payloads_exclude_billing_pii(self):
        scorer = _ScoringLLM()
        original_scorer = gtm_agent._scoring_llm
        gtm_agent._scoring_llm = scorer
        try:
            data_service._PROFILES.clear()
            offering = next(iter(OFFERINGS.values()))
            for prospect_id in PROSPECTS:
                contact = gtm_agent.get_prospect.invoke({"prospect_id": prospect_id})
                profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
                injected_profile = {
                    **profile["prospect_profile"],
                    "billing_qualification": PROSPECTS[prospect_id]["billing_qualification"],
                    "ssn": "123-45-6789",
                    "card_number": "4111111111111111",
                }
                gtm_agent.score_prospect.invoke({
                    "prospect_profile": injected_profile,
                    "offering": offering,
                })
                for result in (contact, profile):
                    self.assertNotIn("billing_qualification", json.dumps(result))
                    self.assertIsNone(re.search(r"\b\d{3}-\d{2}-\d{4}\b", json.dumps(result)))
                    self.assertIsNone(re.search(r"\b\d{16}\b", json.dumps(result)))
            payload = json.dumps(scorer.payloads)
            self.assertNotIn("billing_qualification", payload)
            self.assertIsNone(re.search(r"\b\d{3}-\d{2}-\d{4}\b", payload))
            self.assertIsNone(re.search(r"\b\d{16}\b", payload))
        finally:
            data_service._PROFILES.clear()
            gtm_agent._scoring_llm = original_scorer


if __name__ == "__main__":
    unittest.main()
