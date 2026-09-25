"""
AssureX Root Rule Engine Shim
Exposes the core WarrantyRuleEngine and evaluate_claim function from src.rule_engine.
"""

from src.rule_engine import (
    WarrantyRuleEngine,
    RuleEngineError,
    evaluate_claim,
    parse_date,
    calculate_date_difference_days
)

__all__ = [
    "WarrantyRuleEngine",
    "RuleEngineError",
    "evaluate_claim",
    "parse_date",
    "calculate_date_difference_days"
]

if __name__ == "__main__":
    import os
    import json

    print("Running AssureX Rule Engine from workspace root...")
    policy_path = os.path.join("policies", "laptop_policy.json")
    if os.path.exists(policy_path):
        sample_claim = {
            "claim_id": "CLM-ROOT-DEMO-01",
            "product_category": "Laptop",
            "serial_number": "LNV-IDEAPAD-82H8-9921",
            "invoice_serial_number": "LNV-IDEAPAD-82H8-9921",
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
        res = evaluate_claim(policy_path, sample_claim)
        print(json.dumps(res, indent=2))
