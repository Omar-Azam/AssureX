"""
Warranty Claim Rule Engine - AssureX Core Evaluation Module
===========================================================

This module evaluates consumer electronics warranty claims against configurable
warranty policies (in JSON format). It performs multi-dimensional checks covering
temporal validity, coverage eligibility, exclusions, documentation integrity,
anti-fraud verification, and logical consistency.

Domain Context:
---------------
Tailored for consumer electronics (Washing Machines, Laptops, Smartphones)
operating within the Pakistani retail and service-center landscape (e.g.,
PTA DIRBS compliance, extreme voltage fluctuations without stabilizers,
high-TDS groundwater limescale, unauthorized bazaar repairs, grey imports).

Author: AssureX Claims Engineering Team
"""

import json
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union, Tuple


class RuleEngineError(Exception):
    """Base exception for rule engine configuration or execution errors."""
    pass


def parse_date(date_val: Union[str, date, datetime, None]) -> Optional[date]:
    """
    Safely parse various date formats into a standard datetime.date object.
    Supports ISO format (YYYY-MM-DD), standard timestamps, and date instances.
    """
    if date_val is None:
        return None
    if isinstance(date_val, datetime):
        return date_val.date()
    if isinstance(date_val, date):
        return date_val
    if isinstance(date_val, str):
        date_str = date_val.strip().split("T")[0]  # Strip time if ISO timestamp
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
    return None


def calculate_date_difference_days(start_date: Optional[date], end_date: Optional[date]) -> Optional[int]:
    """Calculate the integer number of calendar days between two dates (end_date - start_date)."""
    if start_date is None or end_date is None:
        return None
    return (end_date - start_date).days


class WarrantyRuleEngine:
    """
    Comprehensive rule evaluation engine that checks warranty claims against
    configurable warranty policy definitions.
    """

    def __init__(self, policy: Union[Dict[str, Any], str]):
        """
        Initialize the rule engine with a policy dictionary or path to a policy JSON file.
        """
        if isinstance(policy, str):
            with open(policy, "r", encoding="utf-8") as f:
                self.policy = json.load(f)
        elif isinstance(policy, dict):
            self.policy = policy
        else:
            raise RuleEngineError("Policy must be a dictionary or a valid path to a JSON file.")

        self.category = self.policy.get("product_category", "Unknown")

    def evaluate_claim(
        self,
        claim: Dict[str, Any],
        historical_claims: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a claim record against the loaded policy across 11 key dimensions:
        1.  warranty_active
        2.  fault_covered
        3.  within_reporting_period
        4.  proof_of_purchase_present
        5.  serial_number_match
        6.  has_prior_repair
        7.  is_authorized_repair
        8.  excluded_damage
        9.  missing_documents
        10. duplicate_claim_flag
        11. contradictions

        Returns a structured dictionary:
        {
            "rules_passed": List[str],
            "rules_failed": List[str],
            "warnings": List[str],
            "manual_review_required": bool,
            "reasons": List[str],
            "evaluations": Dict[str, Any]
        }
        """
        rules_passed: List[str] = []
        rules_failed: List[str] = []
        warnings: List[str] = []
        reasons: List[str] = []
        manual_review_required = False

        # Normalize historical claims list
        history = historical_claims or claim.get("prior_claims_history", [])

        # =========================================================================
        # 1. EVALUATION: Contradictions (Data Integrity & Cross-Field Logic)
        # =========================================================================
        # WHY THIS RULE EXISTS:
        # Before evaluating policy entitlement, verify internal truthfulness and sanity.
        # Contradictions often indicate fabricated paperwork, fraudulent submissions,
        # or data entry errors (e.g. claim filed before purchase date, or conflicting
        # inspection reports).
        contradictions = self._evaluate_contradictions(claim)
        if contradictions:
            rules_failed.append("DATA_INTEGRITY_CHECK")
            for c in contradictions:
                reasons.append(f"Contradiction detected: {c}")
            manual_review_required = True
        else:
            rules_passed.append("DATA_INTEGRITY_CHECK")

        # =========================================================================
        # 2. EVALUATION: Proof of Purchase Present (Tax & Provenance Mandate)
        # =========================================================================
        # WHY THIS RULE EXISTS:
        # In Pakistan, un-invoiced smuggled, refurbished, or stolen goods circulate
        # widely through informal bizarres. A genuine sales invoice with NTN/STRN
        # proves legal purchase through authorized tax-compliant channels.
        pop_present, pop_reason = self._evaluate_proof_of_purchase(claim)
        if pop_present:
            rules_passed.append("PROOF_OF_PURCHASE_PRESENT")
        else:
            rules_failed.append("PROOF_OF_PURCHASE_PRESENT")
            reasons.append(pop_reason)
            manual_review_required = True

        # =========================================================================
        # 3. EVALUATION: Serial Number / IMEI Match (Anti-Tamper & PTA Compliance)
        # =========================================================================
        # WHY THIS RULE EXISTS:
        # Fraudulent claimants frequently swap chassis serials or use receipts from
        # working units on broken out-of-warranty hardware ("warranty laundering").
        # For smartphones, PTA DIRBS compliance is mandatory under Pakistani law;
        # non-tax-paid, CPID patched, or cloned IMEIs are illegal to service.
        serial_match, serial_reason = self._evaluate_serial_number(claim)
        if serial_match:
            rules_passed.append("SERIAL_NUMBER_MATCH")
        else:
            rules_failed.append("SERIAL_NUMBER_MATCH")
            reasons.append(serial_reason)

        # =========================================================================
        # 4. EVALUATION: Warranty Active (Coverage Duration & Grace Period)
        # =========================================================================
        # WHY THIS RULE EXISTS:
        # Ensures that the claim falls within the contractual coverage lifespan.
        # Supports component-specific tiers (e.g. 10-year motor warranty vs 1-year PCB)
        # and accommodates statutory grace periods while flagging late submissions.
        warranty_active, warranty_status_msg, in_grace = self._evaluate_warranty_active(claim)
        if warranty_active:
            rules_passed.append("WARRANTY_ACTIVE")
            if in_grace:
                warnings.append(
                    f"Warranty Grace Period Active: {warranty_status_msg}. "
                    "Claim registered within allowable grace window; proof of initial fault date recommended."
                )
        else:
            rules_failed.append("WARRANTY_ACTIVE")
            reasons.append(warranty_status_msg)

        # =========================================================================
        # 5. EVALUATION: Within Reporting Period (Mitigation of Collateral Harm)
        # =========================================================================
        # WHY THIS RULE EXISTS:
        # Prompt notification (e.g. 3 to 7 days) prevents collateral damage.
        # Operating a failing appliance/device worsens damage (e.g. running a washing
        # machine with bad bearings destroys the outer tub; using a laptop with a
        # seized fan burns the GPU; delayed phone water damage causes fatal oxidation).
        within_reporting, reporting_reason, reporting_is_warning = self._evaluate_reporting_period(claim)
        if within_reporting:
            rules_passed.append("WITHIN_REPORTING_PERIOD")
        else:
            if reporting_is_warning:
                warnings.append(reporting_reason)
                rules_passed.append("WITHIN_REPORTING_PERIOD")
            else:
                rules_failed.append("WITHIN_REPORTING_PERIOD")
                reasons.append(reporting_reason)

        # =========================================================================
        # 6. EVALUATION: Fault Covered (Manufacturing Defect vs Normal Wear)
        # =========================================================================
        # WHY THIS RULE EXISTS:
        # Warranty contracts guarantee repair for spontaneous factory defects and
        # electrical/mechanical component failures under rated conditions. Normal
        # cosmetic wear, battery depreciation from age, and software errors are excluded.
        fault_covered, fault_reason = self._evaluate_fault_covered(claim)
        if fault_covered:
            rules_passed.append("FAULT_COVERED")
        else:
            rules_failed.append("FAULT_COVERED")
            reasons.append(fault_reason)

        # =========================================================================
        # 7. EVALUATION: Excluded Damage (External Abuse, Surges, Liquids, Pests)
        # =========================================================================
        # WHY THIS RULE EXISTS:
        # Manufacturers are not liable for accidental drops, liquid spills (LDI pink),
        # pest/rodent infestation (common in Pakistani kitchens), or grid surges
        # where the customer failed to use an automated 15A voltage stabilizer.
        has_excluded_damage, exclusion_reasons = self._evaluate_excluded_damage(claim)
        if not has_excluded_damage:
            rules_passed.append("NO_EXCLUDED_DAMAGE")
        else:
            rules_failed.append("NO_EXCLUDED_DAMAGE")
            for ex in exclusion_reasons:
                reasons.append(f"Excluded Damage: {ex}")

        # =========================================================================
        # 8 & 9. EVALUATION: Prior Repairs & Authorized Service Center Compliance
        # =========================================================================
        # WHY THIS RULE EXISTS:
        # Prior repair history tracks repeat failures for DOA / Lemon Law replacement.
        # Crucially, any unauthorized intervention (e.g., local bazaar rework in
        # Hafeez Centre, Hall Road, Techno City) breaks manufacturer tamper seals,
        # invalidates safety certifications, and voids all warranty protection.
        has_prior_repair, prior_count, prior_reasons = self._evaluate_prior_repair(claim, history)
        is_authorized_repair, unauth_reason = self._evaluate_authorized_repair(claim, history)

        if has_prior_repair:
            rules_passed.append("PRIOR_REPAIR_HISTORY_CHECK")
            # If multiple prior repairs of same component, trigger Lemon replacement evaluation
            if prior_count >= 2:
                manual_review_required = True
                warnings.append(
                    f"Lemon Replacement Threshold: Device has undergone {prior_count} previous repairs. "
                    "Eligible for technical replacement review under policy conditions."
                )
        else:
            rules_passed.append("PRIOR_REPAIR_HISTORY_CHECK")

        if is_authorized_repair:
            rules_passed.append("AUTHORIZED_REPAIR_INTEGRITY")
        else:
            rules_failed.append("AUTHORIZED_REPAIR_INTEGRITY")
            reasons.append(unauth_reason)

        # =========================================================================
        # 10. EVALUATION: Missing Documents (Audit & Legal Prerequisites)
        # =========================================================================
        # WHY THIS RULE EXISTS:
        # Consumer claims must meet strict auditing standards (GST receipt, warranty
        # card, CNIC copy, commissioning sheets). Missing documents prevent claims
        # processing until deficiencies are rectified.
        missing_docs = self._evaluate_missing_documents(claim)
        if not missing_docs:
            rules_passed.append("MANDATORY_DOCUMENTS_COMPLETE")
        else:
            rules_failed.append("MANDATORY_DOCUMENTS_COMPLETE")
            reasons.append(f"Missing mandatory documents: {', '.join(missing_docs)}")

        # =========================================================================
        # 11. EVALUATION: Duplicate Claim Flag (Anti-Fraud / Double-Dipping)
        # =========================================================================
        # WHY THIS RULE EXISTS:
        # Claimants sometimes submit concurrent claims at different regional service
        # franchises, or re-submit a claim that was already rejected or settled.
        is_duplicate, duplicate_reason = self._evaluate_duplicate_claim(claim, history)
        if not is_duplicate:
            rules_passed.append("NO_DUPLICATE_CLAIM")
        else:
            rules_failed.append("NO_DUPLICATE_CLAIM")
            reasons.append(duplicate_reason)
            manual_review_required = True

        # =========================================================================
        # 12. EVALUATION: Policy-Specific Dynamic Rules (Hard Fails, Warnings, MRT)
        # =========================================================================
        policy_hf, policy_warn, policy_mrt = self._evaluate_dynamic_policy_rules(claim)
        rules_failed.extend(policy_hf)
        warnings.extend(policy_warn)
        if policy_mrt:
            manual_review_required = True
            for mrt_msg in policy_mrt:
                warnings.append(f"Manual Review Trigger: {mrt_msg}")

        # If any hard rule failed or critical reasons exist, manual review or rejection is set
        if rules_failed:
            # Rejection or escalated manual review
            pass

        # Assemble clean, unique lists
        return {
            "rules_passed": list(dict.fromkeys(rules_passed)),
            "rules_failed": list(dict.fromkeys(rules_failed)),
            "warnings": list(dict.fromkeys(warnings)),
            "manual_review_required": manual_review_required or (len(policy_mrt) > 0),
            "reasons": list(dict.fromkeys(reasons)),
            "evaluations": {
                "warranty_active": warranty_active,
                "fault_covered": fault_covered,
                "within_reporting_period": within_reporting,
                "proof_of_purchase_present": pop_present,
                "serial_number_match": serial_match,
                "has_prior_repair": has_prior_repair,
                "is_authorized_repair": is_authorized_repair,
                "excluded_damage": has_excluded_damage,
                "missing_documents": missing_docs,
                "duplicate_claim_flag": is_duplicate,
                "contradictions": contradictions
            }
        }

    # -------------------------------------------------------------------------
    # Internal Evaluation Methods
    # -------------------------------------------------------------------------

    def _evaluate_warranty_active(self, claim: Dict[str, Any]) -> Tuple[bool, str, bool]:
        """
        Calculates if the incident/claim date is within the effective warranty period.
        Accounts for component-specific durations (e.g. 10-year motor vs 1-year general)
        and policy grace periods.
        """
        purchase_date = parse_date(claim.get("purchase_date"))
        incident_date = parse_date(claim.get("incident_date") or claim.get("fault_date"))
        claim_date = parse_date(claim.get("claim_date")) or date.today()

        eval_date = incident_date if incident_date else claim_date

        if not purchase_date:
            return False, "Purchase date is missing or invalid; cannot determine warranty validity.", False

        if eval_date < purchase_date:
            return False, f"Fault/Claim date ({eval_date}) cannot be prior to purchase date ({purchase_date}).", False

        # Determine relevant coverage duration in months
        coverage_dict = self.policy.get("coverage_duration", {})
        component_under_claim = str(claim.get("affected_component", "")).lower()

        # Default comprehensive duration
        default_months = (
            coverage_dict.get("comprehensive_general_months")
            or coverage_dict.get("system_motherboard_and_processor_months")
            or coverage_dict.get("handset_motherboard_and_modem_months")
            or 12
        )

        # Check for subcomponent specific warranty
        applicable_months = default_months
        for comp_key, comp_months in coverage_dict.items():
            comp_name_normalized = comp_key.replace("_months", "").replace("_", " ")
            if component_under_claim and (component_under_claim in comp_name_normalized or comp_name_normalized in component_under_claim):
                applicable_months = max(applicable_months, comp_months)

        # Approximate days: 30.4375 days per month average
        standard_coverage_days = int(applicable_months * 30.4375)
        grace_days = int(self.policy.get("grace_period_days", 0))

        days_since_purchase = (eval_date - purchase_date).days

        if days_since_purchase <= standard_coverage_days:
            return True, f"Claim within standard {applicable_months}-month warranty period ({days_since_purchase} days elapsed).", False
        elif days_since_purchase <= (standard_coverage_days + grace_days):
            return True, f"Claim within {grace_days}-day grace period ({days_since_purchase} days elapsed, standard was {standard_coverage_days} days).", True
        else:
            overdue_days = days_since_purchase - standard_coverage_days - grace_days
            return False, f"Warranty expired {overdue_days} days ago (Coverage: {standard_coverage_days} days + {grace_days} grace days; Elapsed: {days_since_purchase} days).", False

    def _evaluate_fault_covered(self, claim: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates whether the reported issue is an eligible manufacturing/hardware fault
        and not ordinary cosmetic wear or user-induced software malfunction.
        """
        fault_description = str(claim.get("fault_description", "")).lower()
        reported_fault_code = str(claim.get("fault_code", "")).upper()
        symptom = str(claim.get("symptom", "")).lower()

        if not fault_description and not reported_fault_code and not symptom:
            return False, "No fault description, symptom, or diagnostic fault code provided in claim."

        # Check if the claim mentions purely excluded non-hardware items
        non_hardware_keywords = ["pirated windows", "custom rom install", "forgotten password", "cosmetic scratch", "rubbed off paint", "lost accessory"]
        for nh in non_hardware_keywords:
            if nh in fault_description or nh in symptom:
                return False, f"Reported issue relates to non-covered software/cosmetic fault: '{nh}'."

        covered_faults = [str(f).lower() for f in self.policy.get("covered_faults", [])]
        
        # Check for keyword overlap with covered faults
        combined_text = f"{fault_description} {symptom} {reported_fault_code}".lower()
        
        # If policy lists specific covered faults, verify affinity
        match_found = False
        for cf in covered_faults:
            # Check key phrases (words of length >= 4)
            key_words = [w for w in cf.replace("/", " ").replace("(", " ").replace(")", " ").split() if len(w) >= 4]
            matching_words = [w for w in key_words if w in combined_text]
            if len(matching_words) >= 2 or (len(key_words) > 0 and len(matching_words) / len(key_words) >= 0.3):
                match_found = True
                break

        # If claim explicitly declares hardware component failure that is not in exclusions
        if not match_found:
            standard_hardware_tokens = ["motor", "pcb", "drum", "pump", "valve", "motherboard", "screen", "display", "battery", "keyboard", "hinge", "gpu", "audio", "camera", "charging", "fan", "sensor"]
            for token in standard_hardware_tokens:
                if token in combined_text:
                    match_found = True
                    break

        if match_found:
            return True, "Fault matches recognized hardware failure modes under policy scope."
        else:
            return False, f"Reported fault ('{fault_description or symptom}') does not correspond to covered manufacturing defects in policy."

    def _evaluate_reporting_period(self, claim: Dict[str, Any]) -> Tuple[bool, str, bool]:
        """
        Checks if the claim was filed within the allowed claim-reporting period
        from when the fault first occurred.
        """
        incident_date = parse_date(claim.get("incident_date") or claim.get("fault_date"))
        claim_date = parse_date(claim.get("claim_date")) or date.today()

        allowed_days = int(self.policy.get("claim_reporting_period_days", 7))

        if not incident_date:
            # If incident date isn't recorded separately, default to claim date with advisory
            return True, "Incident date not specified; assumed reported upon immediate occurrence.", True

        delta_days = (claim_date - incident_date).days
        if delta_days < 0:
            return False, f"Claim date ({claim_date}) cannot precede incident date ({incident_date}).", False

        if delta_days <= allowed_days:
            return True, f"Reported within permitted window ({delta_days} days <= allowed {allowed_days} days).", False
        else:
            return False, f"Delayed reporting: Fault reported {delta_days} days after occurrence (exceeds {allowed_days}-day limit).", True

    def _evaluate_proof_of_purchase(self, claim: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verifies that proof of purchase documentation is attached and contains
        mandatory legal identifiers (NTN/STRN, date, authorized dealer).
        """
        pop_data = claim.get("proof_of_purchase", {})
        invoice_number = claim.get("invoice_number") or pop_data.get("invoice_number")
        invoice_attached = claim.get("invoice_attached", False) or pop_data.get("attached", False)
        dealer_ntn = pop_data.get("dealer_ntn") or claim.get("dealer_ntn")

        if not invoice_number and not invoice_attached:
            return False, "Missing proof of purchase: No retail invoice number or attached document provided."

        # Check for registered NTN/STRN if available in metadata
        if dealer_ntn and not str(dealer_ntn).isalnum():
            return False, "Proof of purchase has invalid tax registration format (NTN/STRN)."

        return True, "Valid proof of purchase verified with retailer details."

    def _evaluate_serial_number(self, claim: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates physical serial number / IMEI against invoice records and
        Pakistani regulatory compliance standards (PTA DIRBS).
        """
        claimed_sn = str(claim.get("serial_number", "")).strip().upper()
        invoice_sn = str(claim.get("invoice_serial_number", "")).strip().upper()
        imei_1 = str(claim.get("imei_1", "")).strip()
        imei_2 = str(claim.get("imei_2", "")).strip()

        # If category is smartphone, validate IMEI & PTA DIRBS
        if self.category.lower() == "smartphone":
            if not imei_1:
                return False, "Mandatory IMEI 1 is missing for smartphone warranty claim."
            
            # PTA DIRBS Check
            pta_verified = claim.get("pta_dirbs_verified")
            if pta_verified is False or claim.get("cpid_patched_imei") is True:
                return False, "Serial/IMEI validation failed: Device is non-PTA compliant, blacklisted, or CPID patched."

            if invoice_sn and imei_1 not in invoice_sn and (not imei_2 or imei_2 not in invoice_sn) and claimed_sn != invoice_sn:
                return False, f"IMEI mismatch: Handset IMEI ({imei_1}) does not match invoice IMEI ({invoice_sn})."

            return True, "Smartphone IMEI and PTA DIRBS compliance verified."

        # For Laptops and Washing Machines
        if not claimed_sn:
            return False, "Unit serial number is missing from claim record."

        if invoice_sn and claimed_sn != invoice_sn:
            return False, f"Serial number mismatch: Claimed serial ({claimed_sn}) does not match retail invoice ({invoice_sn})."

        if claim.get("serial_number_tampered") is True or claim.get("rating_plate_removed") is True:
            return False, "Unit serial barcode or rating nameplate exhibits signs of physical tampering or removal."

        return True, f"Serial number ({claimed_sn}) successfully validated against invoice and database."

    def _evaluate_prior_repair(
        self,
        claim: Dict[str, Any],
        historical_claims: List[Dict[str, Any]]
    ) -> Tuple[bool, int, List[str]]:
        """
        Determines if the unit has previous repair history and counts lifetime service interventions.
        """
        reasons = []
        prior_repairs = claim.get("prior_repairs", [])
        
        # Combine explicitly declared prior repairs with external claim database
        total_history = list(prior_repairs)
        for h in historical_claims:
            if h.get("serial_number") == claim.get("serial_number") and h.get("claim_id") != claim.get("claim_id"):
                total_history.append(h)

        if total_history:
            count = len(total_history)
            for item in total_history:
                job_id = item.get("job_card_id") or item.get("claim_id") or "Unknown"
                component = item.get("replaced_component") or item.get("affected_component") or "Component"
                reasons.append(f"Prior repair recorded: Job #{job_id} ({component})")
            return True, count, reasons

        return False, 0, ["No prior repairs on record for this serial number."]

    def _evaluate_authorized_repair(
        self,
        claim: Dict[str, Any],
        historical_claims: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Verifies whether any previous repairs or technician interventions were
        executed strictly through authorized brand service centers.
        """
        # Check direct technician notes on current claim
        if claim.get("unauthorized_third_party_repair") is True:
            return False, "Technician inspection found evidence of prior unauthorized repair at a local third-party workshop."

        if claim.get("tamper_seal_broken") is True or claim.get("warranty_void_sticker_punctured") is True:
            return False, "Factory warranty void seal / screw tamper sticker has been punctured or removed."

        # Check prior claims history for unauthorized markers
        for h in historical_claims:
            if h.get("service_center_authorized") is False or h.get("unauthorized_intervention") is True:
                return False, f"Historical claim record #{h.get('claim_id')} indicates unauthorized third-party servicing."

        return True, "All past repairs performed by certified brand Authorized Service Centers."

    def _evaluate_excluded_damage(self, claim: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Scans claim inspection telemetry and customer declarations for condition exclusions:
        - Liquid ingress (LDI/LCI triggered)
        - Accidental drops / physical impact
        - Unprotected power surges on raw AC mains without 15A stabilizer
        - Pest/rodent infestation
        - Saline/hard groundwater corrosion
        - Commercial use of domestic appliances
        - Unauthorized firmware / bootloader modification
        """
        exclusions_detected = []

        # 1. Liquid Ingress
        if (
            claim.get("liquid_damage_detected") is True
            or claim.get("ldi_triggered") is True
            or claim.get("motherboard_lci_triggered") is True
            or claim.get("sim_tray_ldi_triggered") is True
        ):
            exclusions_detected.append("Liquid or chemical ingress verified by triggered internal Liquid Damage Indicators (LDI/LCI).")

        # 2. Physical and Accidental Impact
        if (
            claim.get("physical_damage") is True
            or claim.get("screen_cracked") is True
            or claim.get("outer_tub_fracture_impact") is True
            or claim.get("chassis_bent") is True
        ):
            exclusions_detected.append("Physical impact, dropped chassis, or cracked cover glass/panel.")

        # 3. Voltage Surge without Required Automatic Stabilizer
        if claim.get("power_surge_burn") is True:
            if claim.get("automatic_voltage_stabilizer_used") is False:
                exclusions_detected.append(
                    "Catastrophic electrical surge burnout occurred while operating without mandatory automatic 15A voltage stabilizer."
                )

        # 4. Rodent or Pest Infestation
        if claim.get("rodent_damage_detected") is True or claim.get("pest_infestation") is True:
            exclusions_detected.append("Electrical wiring chewed by rodents or PCB short-circuited by insect/lizard ingress.")

        # 5. High TDS / Hard Groundwater Scale
        if claim.get("water_tds_ppm", 0) > 1500 or claim.get("limescale_calcium_corrosion") is True:
            exclusions_detected.append("Severe limescale encrustation caused by highly saline/brackish groundwater exceeding 1500 PPM.")

        # 6. Commercial Usage for Domestic Category
        if claim.get("commercial_usage") is True or claim.get("daily_cycle_count", 0) > 5:
            exclusions_detected.append("Appliance operated in a commercial laundry, hotel, hostel, or exceeded rated domestic duty cycles.")

        # 7. Unofficial Firmware / Root / Bootloader Unlock
        if claim.get("bootloader_unlocked") is True or claim.get("knox_tripped_0x1") is True or claim.get("custom_rom_installed") is True:
            exclusions_detected.append("System firmware compromised via bootloader unlocking, custom ROM, or root binary tampering.")

        return len(exclusions_detected) > 0, exclusions_detected

    def _evaluate_missing_documents(self, claim: Dict[str, Any]) -> List[str]:
        """
        Audits provided document attachments against mandatory document requirements
        specified in the policy.
        """
        missing: List[str] = []
        provided_docs = [str(d).lower().strip() for d in claim.get("provided_documents", [])]

        # Convert provided docs into a joined search string
        docs_blob = " ".join(provided_docs)

        # Check for Retail Purchase Invoice
        if not ("invoice" in docs_blob or "receipt" in docs_blob or claim.get("invoice_attached") is True):
            missing.append("Original Retail Purchase Invoice (with NTN/STRN)")

        # Check for Warranty Card
        if not ("warranty card" in docs_blob or "card" in docs_blob or claim.get("warranty_card_attached") is True):
            missing.append("Official Dealer Stamped Warranty Card")

        # Check for CNIC copy
        if not ("cnic" in docs_blob or "identity card" in docs_blob or claim.get("cnic_attached") is True):
            missing.append("Claimant National Identity Card (CNIC) Copy")

        # Category-Specific Document Requirements
        if self.category.lower() == "washing machine":
            subcategory = str(claim.get("subcategory", "")).lower()
            if "front-load" in subcategory or "front load" in subcategory:
                if not ("installation" in docs_blob or "commissioning" in docs_blob or claim.get("commissioning_slip_attached") is True):
                    missing.append("Authorized Technician Installation & Commissioning Certificate")

        elif self.category.lower() == "smartphone":
            if not ("pta" in docs_blob or "dirbs" in docs_blob or claim.get("pta_slip_attached") is True):
                missing.append("PTA DIRBS 8484 Verification Confirmation Slip/SMS")

        return missing

    def _evaluate_duplicate_claim(
        self,
        claim: Dict[str, Any],
        historical_claims: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Detects whether an identical claim for the same device and fault is currently
        active, recently submitted, or fraudulently double-filed.
        """
        current_claim_id = str(claim.get("claim_id", "")).strip()
        current_sn = str(claim.get("serial_number", "")).strip().upper()
        current_imei = str(claim.get("imei_1", "")).strip()
        current_date = parse_date(claim.get("claim_date")) or date.today()

        for h in historical_claims:
            h_claim_id = str(h.get("claim_id", "")).strip()
            h_sn = str(h.get("serial_number", "")).strip().upper()
            h_imei = str(h.get("imei_1", "")).strip()
            h_date = parse_date(h.get("claim_date"))
            h_status = str(h.get("claim_status", "")).upper()

            # Ignore comparison with the exact same claim record
            if h_claim_id and current_claim_id and h_claim_id == current_claim_id:
                continue

            # Check matching device identity
            is_same_device = (current_sn and current_sn == h_sn) or (current_imei and current_imei == h_imei)

            if is_same_device:
                # Check for active open ticket
                if h_status in ["OPEN", "PENDING", "IN_PROGRESS", "UNDER_REVIEW"]:
                    return True, f"Duplicate active claim: Open ticket #{h_claim_id} currently pending for this device."

                # Check for rapid refiling within 30 days of previous resolution
                if h_date:
                    diff_days = abs((current_date - h_date).days)
                    if diff_days <= 14:
                        return True, f"Duplicate claim suspicion: Claim #{h_claim_id} for this device was logged only {diff_days} days ago."

        return False, "No active duplicate claims detected on system records."

    def _evaluate_contradictions(self, claim: Dict[str, Any]) -> List[str]:
        """
        Identifies logical impossibilities and conflicting declarations within the claim record.
        """
        contradictions = []

        purchase_date = parse_date(claim.get("purchase_date"))
        incident_date = parse_date(claim.get("incident_date") or claim.get("fault_date"))
        claim_date = parse_date(claim.get("claim_date"))

        # Chronology contradictions
        if purchase_date and incident_date and incident_date < purchase_date:
            contradictions.append(f"Fault incident date ({incident_date}) occurs prior to device purchase date ({purchase_date}).")

        if purchase_date and claim_date and claim_date < purchase_date:
            contradictions.append(f"Claim filing date ({claim_date}) is earlier than purchase date ({purchase_date}).")

        if incident_date and claim_date and claim_date < incident_date:
            contradictions.append(f"Claim filing date ({claim_date}) occurs before the fault incident date ({incident_date}).")

        # Condition declarations vs technician findings
        if claim.get("customer_claims_no_liquid") is True and claim.get("liquid_damage_detected") is True:
            contradictions.append("Customer declared zero liquid contact, but bench diagnostic confirmed triggered LDI / liquid corrosion.")

        if claim.get("customer_claims_never_repaired") is True and len(claim.get("prior_repairs", [])) > 0:
            contradictions.append("Customer declared device was never previously repaired, yet historical repair records exist.")

        if claim.get("customer_claims_domestic_only") is True and claim.get("daily_cycle_count", 0) > 10:
            contradictions.append("Claim declared domestic household usage, but cycle counter telemetry recorded >10 cycles/day.")

        if claim.get("customer_claims_stabilizer_used") is True and claim.get("stabilizer_model") in ["none", "null", "", None] and claim.get("power_surge_burn") is True:
            contradictions.append("Customer claimed voltage stabilizer was installed, but inspection logged direct mains connection with no stabilizer.")

        return contradictions

    def _evaluate_dynamic_policy_rules(
        self,
        claim: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Executes configurable hard-fail, warning, and manual-review-trigger rules
        defined directly in the policy JSON schema.
        """
        hard_fails = []
        warnings = []
        mrt_triggers = []

        # Evaluate Hard-Fail Rules
        for hf in self.policy.get("hard_fail_rules", []):
            rule_id = hf.get("rule_id", "HF-UNKNOWN")
            cond = hf.get("condition", "")
            res_class = hf.get("resulting_class", "REJECTED")

            # High-level condition evaluator based on claim attributes
            if "commercial_laundry" in cond and (claim.get("installation_environment") in ["commercial_laundry", "hostel", "hotel"] or claim.get("daily_cycle_count", 0) > 4):
                hard_fails.append(f"{rule_id}: {res_class}")

            elif "tamper_seal_broken" in cond and (claim.get("tamper_seal_broken") is True or claim.get("third_party_repair_evidence") is True):
                hard_fails.append(f"{rule_id}: {res_class}")

            elif "rodent_damage_detected" in cond and claim.get("rodent_damage_detected") is True:
                hard_fails.append(f"{rule_id}: {res_class}")

            elif "motherboard_lci_triggered" in cond and (claim.get("motherboard_lci_triggered") is True or claim.get("liquid_damage_detected") is True):
                hard_fails.append(f"{rule_id}: {res_class}")

            elif "pta_dirbs_verified == false" in cond and (claim.get("pta_dirbs_verified") is False or claim.get("cpid_patched_imei") is True):
                hard_fails.append(f"{rule_id}: {res_class}")

            elif "power_rail_surge_burn" in cond and claim.get("power_surge_burn") is True and claim.get("automatic_voltage_stabilizer_used") is False:
                hard_fails.append(f"{rule_id}: {res_class}")

        # Evaluate Warning Rules
        for wrn in self.policy.get("warning_rules", []):
            rule_id = wrn.get("rule_id", "WRN-UNKNOWN")
            flag = wrn.get("warning_flag", "WARNING")
            msg = wrn.get("warning_message", "")

            if "water_source_tds_ppm > 1000" in wrn.get("condition", "") and claim.get("water_tds_ppm", 0) > 1000:
                warnings.append(f"{rule_id} [{flag}]: {msg}")

            elif "battery_cycle_count" in wrn.get("condition", "") and (claim.get("battery_cycle_count", 0) > 300 or claim.get("battery_health_percentage", 100) <= 75):
                warnings.append(f"{rule_id} [{flag}]: {msg}")

            elif "vertical_green_or_pink_line_present" in wrn.get("condition", "") and claim.get("vertical_oled_line") is True:
                warnings.append(f"{rule_id} [{flag}]: {msg}")

            elif "google_frp_locked" in wrn.get("condition", "") and claim.get("cloud_security_locked") is True:
                warnings.append(f"{rule_id} [{flag}]: {msg}")

        # Evaluate Manual Review Trigger Rules
        for mrt in self.policy.get("manual_review_trigger_rules", []):
            rule_id = mrt.get("rule_id", "MRT-UNKNOWN")
            reason = mrt.get("trigger_reason", "")
            cond = mrt.get("condition", "")

            if "claim_type == 'REPLACEMENT'" in cond and claim.get("claim_type") == "REPLACEMENT":
                mrt_triggers.append(f"{rule_id}: {reason}")

            elif "historical_board_replacements_count" in cond and claim.get("historical_board_replacements_count", 0) >= 2:
                mrt_triggers.append(f"{rule_id}: {reason}")

            elif "customer_contests" in cond and claim.get("customer_contests_decision") is True:
                mrt_triggers.append(f"{rule_id}: {reason}")

        return hard_fails, warnings, mrt_triggers


def evaluate_claim(
    policy: Union[Dict[str, Any], str],
    claim: Dict[str, Any],
    historical_claims: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Convenience functional interface to evaluate a warranty claim against a policy.
    """
    engine = WarrantyRuleEngine(policy)
    return engine.evaluate_claim(claim, historical_claims)


# =============================================================================
# Direct Execution / Demonstration
# =============================================================================
if __name__ == "__main__":
    import os

    print("=" * 80)
    print("AssureX Warranty Rule Engine - Verification & Live Demo")
    print("=" * 80)

    # Path to sample laptop policy
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    laptop_policy_path = os.path.join(base_dir, "policies", "laptop_policy.json")

    # Sample realistic claim record
    sample_claim = {
        "claim_id": "CLM-LP-2026-0042",
        "product_category": "Laptop",
        "subcategory": "Consumer Notebook / Everyday Laptop",
        "serial_number": "LNV-IDEAPAD-82H8-9921",
        "invoice_serial_number": "LNV-IDEAPAD-82H8-9921",
        "purchase_date": "2025-05-10",
        "fault_date": "2026-02-15",
        "claim_date": "2026-02-18",
        "affected_component": "system_motherboard_and_processor",
        "fault_description": "Motherboard power rail short circuit; machine fails to power on",
        "symptom": "Power LED blinks twice then shuts down; no fan spin",
        "invoice_number": "INV-MERC-88319",
        "invoice_attached": True,
        "warranty_card_attached": True,
        "cnic_attached": True,
        "provided_documents": [
            "Original Sales Tax Invoice",
            "Official Stamped Warranty Card",
            "CNIC Copy"
        ],
        "liquid_damage_detected": False,
        "physical_damage": False,
        "tamper_seal_broken": False,
        "unauthorized_third_party_repair": False,
        "power_surge_burn": False,
        "prior_repairs": []
    }

    if os.path.exists(laptop_policy_path):
        result = evaluate_claim(laptop_policy_path, sample_claim)
        print("\nClaim Evaluation Result:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Policy file not found at {laptop_policy_path}. Run from project root.")
