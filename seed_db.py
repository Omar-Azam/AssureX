"""
AssureX Comprehensive Database Seeding Script
=============================================
Populates database with realistic records matching all SRS specifications:
1. 4 Core Roles: Customer, Service Center Employee, Claim Reviewer, Administrator.
2. Products with diverse warranty states: Active, Expired, and Expiring Soon / Boundary Date.
3. Claims covering all 11 required demo scenarios:
   - Valid claim
   - Invalid claim
   - Manual-review claim
   - Expired-warranty claim
   - Missing-document claim
   - Duplicate claim (with prior reference claim and SHA-256 collision)
   - Contradictory claim
   - Serial-mismatch claim
   - Unauthorized-repair claim
   - Boundary-date claim
   - Model-disagreement claim
4. Complete supporting records: Documents with SHA-256 hashes, Repair History,
   Multimodal Predictions, Reviewer Actions, and System Audit Logs.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, date, timedelta

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.database import Base, engine, SessionLocal
from backend.models import (
    User, Product, Warranty, Claim, Document, RepairHistory,
    Prediction, Review, AuditLog
)
from backend.auth import hash_password
from claim_metrics import compute_claim_metrics, parse_date


def compute_sha256(text: str) -> str:
    """Helper to compute deterministic SHA-256 hash for document evidence."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seed_database():
    """Initializes schema and populates demo-ready database records."""
    print("=" * 75)
    print(" AssureX Comprehensive Database Seeder - SRS Demo Dataset")
    print("=" * 75)

    # 1. Cleanly wipe and re-initialize schema tables
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("[+] Clean database reset: all existing tables dropped and schema re-initialized.")

    db = SessionLocal()
    try:
        # ----------------------------------------------------------------------
        # 2. Seed Users across all 4 roles
        # ----------------------------------------------------------------------
        user_definitions = [
            {
                "username": "admin",
                "email": "admin@assurex.com",
                "password": "Admin@12345",
                "full_name": "System Administrator",
                "role": "admin"
            },
            {
                "username": "reviewer",
                "email": "reviewer@assurex.com",
                "password": "Reviewer@12345",
                "full_name": "Senior Claims Assessor",
                "role": "claim_reviewer"
            },
            {
                "username": "service_center",
                "email": "service@assurex.com",
                "password": "Service@12345",
                "full_name": "TechnoCity Authorized Service Hub",
                "role": "service_center"
            },
            {
                "username": "customer",
                "email": "customer@assurex.com",
                "password": "Customer@12345",
                "full_name": "Ali Khan",
                "role": "customer"
            }
        ]

        users = {}
        for u_data in user_definitions:
            user = db.query(User).filter(User.username == u_data["username"]).first()
            if not user:
                user = User(
                    username=u_data["username"],
                    email=u_data["email"],
                    hashed_password=hash_password(u_data["password"]),
                    full_name=u_data["full_name"],
                    role=u_data["role"],
                    is_active=True
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                print(f"[+] Created User ({user.role}): {user.username} / {u_data['password']}")
            else:
                users[user.role] = user
                print(f"[i] User already exists: {user.username} ({user.role})")
            users[user.role] = user

        cust_user = users["customer"]
        reviewer_user = users["claim_reviewer"]
        service_user = users["service_center"]
        admin_user = users["admin"]

        # ----------------------------------------------------------------------
        # 3. Seed Reference Baseline Prior Claim for Duplicate Collision
        # ----------------------------------------------------------------------
        dupe_serial = "HP-PAV-15-77291-KHI"
        prior_prod = db.query(Product).filter(Product.serial_number == dupe_serial).first()
        if not prior_prod:
            prior_prod = Product(
                product_category="Laptop",
                brand="HP",
                product_name="HP Pavilion 15-eg3000",
                model_number="15-EG3025TX",
                serial_number=dupe_serial,
                purchase_price=165000.0,
                purchase_date=date(2025, 7, 10),
                retailer="UniCenter Computer Market Karachi",
                user_id=cust_user.id
            )
            db.add(prior_prod)
            db.commit()
            db.refresh(prior_prod)

            w_met = compute_claim_metrics(prior_prod.purchase_date, date.today(), 12)
            db.add(Warranty(
                product_id=prior_prod.id,
                warranty_duration_months=12,
                warranty_start_date=prior_prod.purchase_date,
                warranty_expiry_date=parse_date(w_met["warranty_expiry_date"]),
                warranty_status=w_met["warranty_status"],
                terms_conditions="Standard manufacturer 1-year limited parts and labor warranty."
            ))
            db.commit()

        prior_claim = db.query(Claim).filter(Claim.claim_id == "CLM-2026-00042").first()
        shared_invoice_hash = compute_sha256("HP-INVOICE-ORIGINAL-77291-PAID")

        if not prior_claim:
            prior_claim = Claim(
                claim_id="CLM-2026-00042",
                user_id=cust_user.id,
                product_id=prior_prod.id,
                claim_submission_date=date(2026, 1, 22),
                fault_occurrence_date=date(2026, 1, 20),
                fault_description="Original prior claim: Motherboard fails to boot, diagnostic amber light blinks 3 times.",
                damage_type="Motherboard Power Circuit Failure",
                status="Approved",
                prior_replacement=False,
                repair_history="0 repairs"
            )
            db.add(prior_claim)
            db.commit()
            db.refresh(prior_claim)

            # Prior document
            db.add(Document(
                claim_id=prior_claim.id,
                file_name="Invoice_HP_Pavilion_Original.pdf",
                file_path=f"uploads/CLM-2026-00042/Invoice_HP_Pavilion_Original.pdf",
                file_type="receipt",
                file_size=142080,
                sha256_hash=shared_invoice_hash,
                is_duplicate=False
            ))
            db.commit()
            print("[+] Seeded Baseline Claim CLM-2026-00042 for Duplicate Collision Tracking.")

        # ----------------------------------------------------------------------
        # 4. Seed Required 11 Demo Scenarios from sample_claims/
        # ----------------------------------------------------------------------
        demo_files = [
            ("valid_claim.json", "Approved", "Consensus Strong Match. Verified genuine purchase invoice, clean serial match, and active coverage."),
            ("invalid_claim.json", "Rejected", "Excluded damage condition: Severe liquid contact detected on motherboard. Violates Section 4.2 warranty terms."),
            ("manual_review_claim.json", "Manual Review", "Lemon law review trigger: Unit has experienced 3 consecutive motor stator failures. Supervisor review required."),
            ("expired_warranty_claim.json", "Rejected", "Warranty physically expired 365 days prior to claim submission date."),
            ("missing_document_claim.json", "Manual Review", "Missing required proof of purchase invoice and official stamped dealer warranty card."),
            ("duplicate_claim.json", "Rejected", "Duplicate claim rejected: Identical invoice SHA-256 hash collision and duplicate defect claim against CLM-2026-00042."),
            ("contradictory_claim.json", "Manual Review", "Integrity contradiction flagged: Claim filing date precedes product purchase date by 156 days."),
            ("serial_mismatch_claim.json", "Manual Review", "Hardware serial mismatch: Invoice serial does not match physical chassis barcode label."),
            ("unauthorized_repair_claim.json", "Rejected", "Unauthorized modification: Hardware opened and repaired at unauthorized third-party shop with broken tamper seals."),
            ("boundary_date_claim.json", "Approved", "Boundary coverage verified: Claim submitted on the exact calendar day of warranty expiration (0 days remaining)."),
            ("model_disagreement_claim.json", "Manual Review", "Multimodal model disagreement: Tabular ML predicted Likely Valid while Visual GTM classifier predicted Likely Invalid.")
        ]

        sample_dir = _PROJECT_ROOT / "sample_claims"

        for filename, expected_status, review_notes in demo_files:
            file_path = sample_dir / filename
            if not file_path.is_file():
                print(f"[!] Warning: Fixture file {filename} not found in sample_claims/.")
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            claim_id = data.get("claim_id")
            existing_clm = db.query(Claim).filter(Claim.claim_id == claim_id).first()
            if existing_clm:
                print(f"[i] Demo claim {claim_id} already exists. Skipping.")
                continue

            # 4.1 Create Product
            prod_serial = data.get("serial_number", f"SN-{claim_id}")
            product = db.query(Product).filter(Product.serial_number == prod_serial).first()
            purch_dt = parse_date(data.get("purchase_date", "2025-06-15"))

            if not product:
                product = Product(
                    product_category=data.get("product_category", "Smartphone"),
                    brand=data.get("brand", "Samsung"),
                    product_name=data.get("model_name", data.get("product_name", "AssureX Device")),
                    model_number=data.get("model_number", "MOD-DEFAULT"),
                    serial_number=prod_serial,
                    purchase_price=float(data.get("purchase_price", 120000.0)),
                    purchase_date=purch_dt,
                    retailer=data.get("retailer", "Official MegaStore"),
                    user_id=cust_user.id
                )
                db.add(product)
                db.commit()
                db.refresh(product)

            # 4.2 Create Warranty with exact calendar-month arithmetic
            duration_months = int(data.get("warranty_duration_months", 12))
            metrics = compute_claim_metrics(purch_dt, date.today(), duration_months)

            # Boundary date special handling
            if "boundary" in filename:
                exp_dt = purch_dt + timedelta(days=365)
                w_status = "Active"
            elif "expired" in filename:
                exp_dt = parse_date(metrics["warranty_expiry_date"])
                w_status = "Expired"
            else:
                exp_dt = parse_date(metrics["warranty_expiry_date"])
                w_status = metrics["warranty_status"]

            warranty = db.query(Warranty).filter(Warranty.product_id == product.id).first()
            if not warranty:
                warranty = Warranty(
                    product_id=product.id,
                    warranty_duration_months=duration_months,
                    warranty_start_date=purch_dt,
                    warranty_expiry_date=exp_dt,
                    warranty_status=w_status,
                    terms_conditions="Manufacturer official policy warranty covering electrical and mechanical defects."
                )
                db.add(warranty)
                db.commit()

            # 4.3 Create Claim
            fault_dt = parse_date(data.get("fault_date", "2026-02-10")) or date.today()
            claim_dt = parse_date(data.get("claim_date", "2026-02-15")) or date.today()

            claim = Claim(
                claim_id=claim_id,
                user_id=cust_user.id,
                product_id=product.id,
                claim_submission_date=claim_dt,
                fault_occurrence_date=fault_dt,
                fault_description=data.get("fault_description", "Reported hardware failure."),
                damage_type=data.get("damage_type", "Display Panel Defect"),
                status=expected_status,
                prior_replacement=data.get("prior_replacement", False),
                repair_history=f"{len(data.get('prior_repairs', []))} repairs"
            )
            db.add(claim)
            db.commit()
            db.refresh(claim)

            # 4.4 Create Documents
            is_dupe_claim = "duplicate" in filename
            docs_list = data.get("provided_documents", ["Original Retail Purchase Invoice", "Warranty Card"])

            for idx, doc_title in enumerate(docs_list):
                if is_dupe_claim and idx == 0:
                    doc_hash = shared_invoice_hash
                    is_dupe = True
                    dupe_ref = prior_claim.documents[0].id if prior_claim.documents else None
                else:
                    doc_hash = compute_sha256(f"{claim_id}_{doc_title}_{idx}")
                    is_dupe = False
                    dupe_ref = None

                doc_ext = ".pdf" if "invoice" in doc_title.lower() or "card" in doc_title.lower() else ".jpg"
                file_type = "receipt" if "invoice" in doc_title.lower() else ("warranty_card" if "warranty" in doc_title.lower() else "product_image")

                db.add(Document(
                    claim_id=claim.id,
                    file_name=f"{doc_title.replace(' ', '_')}{doc_ext}",
                    file_path=f"uploads/{claim_id}/{doc_title.replace(' ', '_')}{doc_ext}",
                    file_type=file_type,
                    file_size=185200 + (idx * 15400),
                    sha256_hash=doc_hash,
                    is_duplicate=is_dupe,
                    duplicate_of_document_id=dupe_ref
                ))
            db.commit()

            # 4.5 Create Repair History records if present
            prior_repairs = data.get("prior_repairs", [])
            for rep in prior_repairs:
                r_date = parse_date(rep.get("repair_date", "2025-10-15")) or date.today()
                is_auth = rep.get("authorized_service_center", True)
                center_name = rep.get("service_center_name", "Official Service Center")
                db.add(RepairHistory(
                    product_id=product.id,
                    claim_id=claim.id,
                    repair_date=r_date,
                    repair_center=center_name,
                    is_authorized=is_auth,
                    repair_cost=18500.0 if not is_auth else 0.0,
                    fault_repaired=rep.get("repaired_component", "Component Replacement"),
                    notes="Third-party service kiosk" if not is_auth else "Authorized warranty repair"
                ))
            db.commit()

            # 4.6 Create Multimodal Prediction Record
            pred_class = data.get("expected_decision", "Manual Review Required")
            py_conf = 0.94 if pred_class == "Likely Valid" else (0.89 if pred_class == "Likely Invalid" else 0.58)
            gtm_conf = 0.91 if pred_class == "Likely Valid" else (0.87 if pred_class == "Likely Invalid" else 0.54)
            consistency = "Strong Match" if pred_class in ["Likely Valid", "Likely Invalid"] else "Acceptable Match"

            if "disagreement" in filename:
                py_class = "Valid Claim"
                gtm_class = "Invalid Claim"
                py_conf, gtm_conf = 0.92, 0.77
                consistency = "Model Disagreement"
            else:
                py_class = "Valid Claim" if pred_class == "Likely Valid" else ("Invalid Claim" if pred_class == "Likely Invalid" else "Manual Review")
                gtm_class = py_class

            py_pred = {
                "predicted_class": py_class,
                "confidence": py_conf,
                "confidence_valid": py_conf if py_class == "Valid Claim" else 0.05,
                "confidence_invalid": py_conf if py_class == "Invalid Claim" else 0.05,
                "confidence_manual_review": py_conf if py_class == "Manual Review" else 0.05
            }
            gtm_pred = {
                "predicted_class": gtm_class,
                "confidence": gtm_conf,
                "confidence_valid": gtm_conf if gtm_class == "Valid Claim" else 0.08,
                "confidence_invalid": gtm_conf if gtm_class == "Invalid Claim" else 0.08,
                "confidence_manual_review": gtm_conf if gtm_class == "Manual Review" else 0.08
            }
            rule_pred = {
                "rules_passed": ["SERIAL_NUMBER_MATCH", "WARRANTY_ACTIVE"] if "invalid" not in filename and "expired" not in filename else [],
                "rules_failed": data.get("expected_failed_rules", []) if "invalid" in filename or "expired" in filename else [],
                "manual_review_required": expected_status == "Manual Review",
                "contradictions": ["Filing date precedes purchase date"] if "contradictory" in filename else []
            }
            dec_pred = {
                "final_decision": pred_class,
                "model_consistency_status": consistency,
                "confidence_difference": round(abs(py_conf - gtm_conf), 4),
                "decision_explanation": {
                    "summary": review_notes,
                    "factors_supporting": [review_notes],
                    "factors_opposing": [],
                    "contradictions": rule_pred["contradictions"],
                    "additional_evidence_needed": [] if expected_status == "Approved" else ["Sales invoice with NTN", "Physical unit inspection"]
                }
            }

            db.add(Prediction(
                claim_id=claim.id,
                final_decision=pred_class,
                model_consistency_status=consistency,
                confidence_difference=round(abs(py_conf - gtm_conf), 4),
                python_prediction=json.dumps(py_pred),
                gtm_prediction=json.dumps(gtm_pred),
                rule_engine_result=json.dumps(rule_pred),
                decision_engine_result=json.dumps(dec_pred)
            ))
            db.commit()

            # 4.7 Create Reviewer Record for Adjudicated Claims
            if expected_status in ["Approved", "Rejected"]:
                db.add(Review(
                    claim_id=claim.id,
                    reviewer_id=reviewer_user.id,
                    action="approve" if expected_status == "Approved" else "reject",
                    decision_notes=review_notes,
                    overridden_decision=None
                ))
                db.commit()

            # 4.8 Log Audit Trail Entry
            db.add(AuditLog(
                user_id=cust_user.id,
                action="CLAIM_SUBMISSION",
                entity_type="Claim",
                entity_id=claim.claim_id,
                details=f"Demo claim {claim.claim_id} seeded into database ({expected_status})",
                ip_address="127.0.0.1"
            ))
            db.commit()

            print(f"[+] Seeded Demo Scenario ({filename}): {claim.claim_id} -> {expected_status}")

        # ----------------------------------------------------------------------
        # 5. Seed System Audit Initialization Log
        # ----------------------------------------------------------------------
        db.add(AuditLog(
            user_id=admin_user.id,
            action="SYSTEM_INITIALIZATION",
            entity_type="System",
            entity_id="ROOT",
            details="All 11 SRS demo claims and role accounts seeded successfully.",
            ip_address="127.0.0.1"
        ))
        db.commit()

        print("=" * 75)
        print(" SEEDING COMPLETED SUCCESSFULLY!")
        print(" Demo Database populated with 4 users, products, warranties, and 11 demo claims.")
        print("=" * 75)

    except Exception as exc:
        db.rollback()
        print(f"[ERROR] Database seeding encountered an error: {exc}")
        raise exc
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
