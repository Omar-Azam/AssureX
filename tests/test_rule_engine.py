"""
Unit Tests for AssureX Warranty Rule Engine
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from src.rule_engine import WarrantyRuleEngine, evaluate_claim


class TestWarrantyRuleEngine(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.laptop_policy_path = os.path.join(self.base_dir, "policies", "laptop_policy.json")
        self.washing_machine_policy_path = os.path.join(self.base_dir, "policies", "washing_machine_policy.json")
        self.smartphone_policy_path = os.path.join(self.base_dir, "policies", "smartphone_policy.json")

    def test_laptop_valid_claim(self):
        claim = {
            "claim_id": "CLM-LP-VALID",
            "product_category": "Laptop",
            "serial_number": "LNV-88219-PK",
            "invoice_serial_number": "LNV-88219-PK",
            "purchase_date": "2025-06-01",
            "fault_date": "2026-01-10",
            "claim_date": "2026-01-12",
            "affected_component": "system_motherboard_and_processor",
            "fault_description": "Motherboard circuit failure with MOSFET short",
            "invoice_number": "INV-12345",
            "invoice_attached": True,
            "warranty_card_attached": True,
            "cnic_attached": True,
            "provided_documents": ["Original Sales Tax Invoice", "Official Stamped Warranty Card", "CNIC Copy"],
            "liquid_damage_detected": False,
            "physical_damage": False,
            "tamper_seal_broken": False,
            "unauthorized_third_party_repair": False
        }
        res = evaluate_claim(self.laptop_policy_path, claim)
        self.assertIn("WARRANTY_ACTIVE", res["rules_passed"])
        self.assertIn("FAULT_COVERED", res["rules_passed"])
        self.assertIn("SERIAL_NUMBER_MATCH", res["rules_passed"])
        self.assertEqual(len(res["rules_failed"]), 0)
        self.assertFalse(res["manual_review_required"])

    def test_laptop_liquid_damage_exclusion(self):
        claim = {
            "claim_id": "CLM-LP-LIQUID",
            "product_category": "Laptop",
            "serial_number": "LNV-88219-PK",
            "invoice_serial_number": "LNV-88219-PK",
            "purchase_date": "2025-06-01",
            "fault_date": "2026-01-10",
            "claim_date": "2026-01-12",
            "affected_component": "system_motherboard_and_processor",
            "fault_description": "Device dead after water spill on keyboard",
            "motherboard_lci_triggered": True,
            "liquid_damage_detected": True,
            "invoice_number": "INV-12345",
            "invoice_attached": True,
            "provided_documents": ["Invoice", "Warranty Card", "CNIC"]
        }
        res = evaluate_claim(self.laptop_policy_path, claim)
        self.assertIn("NO_EXCLUDED_DAMAGE", res["rules_failed"])
        self.assertTrue(res["evaluations"]["excluded_damage"])

    def test_smartphone_non_pta_rejection(self):
        claim = {
            "claim_id": "CLM-SP-NONPTA",
            "product_category": "Smartphone",
            "serial_number": "S24U-9921",
            "imei_1": "358921094829102",
            "invoice_serial_number": "358921094829102",
            "pta_dirbs_verified": False,
            "cpid_patched_imei": True,
            "purchase_date": "2025-10-01",
            "fault_date": "2026-01-05",
            "claim_date": "2026-01-06",
            "affected_component": "logic_board",
            "fault_description": "Network drop issue",
            "invoice_number": "INV-PTA-01",
            "invoice_attached": True,
            "provided_documents": ["Invoice", "Warranty Card", "CNIC"]
        }
        res = evaluate_claim(self.smartphone_policy_path, claim)
        self.assertIn("SERIAL_NUMBER_MATCH", res["rules_failed"])
        self.assertFalse(res["evaluations"]["serial_number_match"])

    def test_washing_machine_unprotected_surge(self):
        claim = {
            "claim_id": "CLM-WM-SURGE",
            "product_category": "Washing Machine",
            "serial_number": "WM-INV-4412",
            "invoice_serial_number": "WM-INV-4412",
            "purchase_date": "2025-03-01",
            "fault_date": "2026-01-15",
            "claim_date": "2026-01-17",
            "affected_component": "inverter_control_pcb",
            "fault_description": "Inverter PCB burned after voltage spike",
            "power_surge_burn": True,
            "automatic_voltage_stabilizer_used": False,
            "invoice_number": "INV-WM-99",
            "invoice_attached": True,
            "provided_documents": ["Invoice", "Warranty Card", "CNIC"]
        }
        res = evaluate_claim(self.washing_machine_policy_path, claim)
        self.assertIn("NO_EXCLUDED_DAMAGE", res["rules_failed"])
        self.assertTrue(res["evaluations"]["excluded_damage"])

    def test_contradiction_detection(self):
        claim = {
            "claim_id": "CLM-CONTRADICTION",
            "product_category": "Laptop",
            "serial_number": "SN-100",
            "invoice_serial_number": "SN-100",
            "purchase_date": "2026-03-01",
            "fault_date": "2026-01-01",  # Fault before purchase date!
            "claim_date": "2026-01-05",
            "customer_claims_no_liquid": True,
            "liquid_damage_detected": True,  # Conflicting technician report!
            "invoice_number": "INV-001",
            "invoice_attached": True,
            "provided_documents": ["Invoice", "Warranty Card", "CNIC"]
        }
        res = evaluate_claim(self.laptop_policy_path, claim)
        self.assertIn("DATA_INTEGRITY_CHECK", res["rules_failed"])
        self.assertTrue(len(res["evaluations"]["contradictions"]) >= 2)
        self.assertTrue(res["manual_review_required"])

    def test_duplicate_claim_detection(self):
        current_claim = {
            "claim_id": "CLM-DUP-02",
            "serial_number": "SN-REPEAT-01",
            "invoice_serial_number": "SN-REPEAT-01",
            "purchase_date": "2025-05-01",
            "claim_date": "2026-01-10",
            "invoice_number": "INV-DUP",
            "invoice_attached": True,
            "provided_documents": ["Invoice", "Warranty Card", "CNIC"]
        }
        history = [
            {
                "claim_id": "CLM-DUP-01",
                "serial_number": "SN-REPEAT-01",
                "claim_date": "2026-01-05",
                "claim_status": "OPEN"
            }
        ]
        res = evaluate_claim(self.laptop_policy_path, current_claim, historical_claims=history)
        self.assertIn("NO_DUPLICATE_CLAIM", res["rules_failed"])
        self.assertTrue(res["evaluations"]["duplicate_claim_flag"])
        self.assertTrue(res["manual_review_required"])


if __name__ == "__main__":
    unittest.main()
