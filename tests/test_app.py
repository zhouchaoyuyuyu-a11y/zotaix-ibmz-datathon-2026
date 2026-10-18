import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import build_brief, private_text_redacted, EVENT_CODE

class BriefTests(unittest.TestCase):
    def test_output_has_event_code(self):
        self.assertEqual(build_brief("A warm romantic anniversary bottle gift")["event_code"], EVENT_CODE)
    def test_structured_patch_bounded(self):
        p = build_brief("A tall slim vessel for a business thank you")["editable_design_patch"]
        self.assertLessEqual(p["height_ratio"], 1.6)
        self.assertGreaterEqual(p["waist_indent"], 0)
    def test_contact_redacted(self):
        x = private_text_redacted("Contact sam@example.com and +1 202 555 0123")
        self.assertNotIn("sam@example.com", x)
        self.assertNotIn("202 555", x)
    def test_short_requests_rejected(self):
        with self.assertRaises(ValueError):
            build_brief("hi")
    def test_explicit_human_review(self):
        x = build_brief("Handcrafted heritage porcelain gift for a wedding")
        self.assertGreater(len(x["human_review_required"]), 0)
    def test_no_learned_customer_data(self):
        x = build_brief("Minimal gift for an office mentor")
        self.assertEqual(x["privacy"]["data_origin"], "synthetic training examples")

if __name__ == "__main__":
    unittest.main()
