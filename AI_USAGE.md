# AssureX Claim Engine — AI Usage & Prompt Log
### Record of AI-Assisted Engineering, Prompt Trajectory & Human Verification

**Document Version:** 1.0.0  
**Effective Date:** 2026-09-27  
**Compliance Mandate:** Transparent record of AI tool usage, engineering intent, modules affected, human oversight, and testing performed.

---

## Overview

During the design, implementation, debugging, and verification of the **AssureX Claim Engine**, AI coding assistance tools were utilized to accelerate development while adhering to software engineering best practices. In accordance with project evaluation requirements, this document logs every major AI interaction, formatted systematically under the required five-part structure:
1. **Tool Name**
2. **Purpose**
3. **Modules Affected**
4. **Changes Made by Me (Human Oversight & Prompts)**
5. **Testing I Performed**

---

## Log of AI Prompts & Engineering Interactions

### Entry 1: Google Teachable Machine (GTM) Classifier Module
* **Tool Name:** Antigravity AI (Google DeepMind)
* **Purpose:** Completely rewrite `gtm_classifier.py` to load the real Keras MobileNet model (`model/gtm_model/keras_model.h5`) and class mappings (`model/gtm_model/labels.txt`), preprocess input images to $224 \times 224$ RGB, normalize pixel values to $[-1.0, 1.0]$, and output genuine 3-class confidence scores without synthetic fallbacks.
* **Modules Affected:**
  * [`gtm_classifier.py`](file:///c:/Users/omar/Desktop/Code/AssureX/gtm_classifier.py)
  * `model/gtm_model/keras_model.h5`
  * `model/gtm_model/labels.txt`
* **Changes Made by Me:**
  * Explicitly prohibited any dummy fallback logic or fabricated confidence numbers.
  * Mandated that if TensorFlow is not installed, the model file is missing, or weight loading fails, the function must raise a descriptive `RuntimeError` and halt immediately.
  * Added a standalone command-line interface (CLI) at the bottom allowing image evaluation directly from the terminal.
* **Testing I Performed:**
  * Tested the CLI with valid Claim Summary Cards in `claim_cards/test/` to verify probability output format.
  * Verified that non-existent image paths immediately trigger descriptive exceptions.

---

### Entry 2: Multimodal Decision Engine & Threshold Configuration
* **Tool Name:** Antigravity AI (Google DeepMind)
* **Purpose:** Develop `decision_engine.py` with `final_claim_decision()` to calculate confidence differences, classify model consistency tiers (Strong Match, Acceptable Match, Weak Match, Model Disagreement, Uncertain Result), arbitrate the final outcome (`Likely Valid`, `Likely Invalid`, `Manual Review Required`), and output transparent natural language explanations.
* **Modules Affected:**
  * [`decision_engine.py`](file:///c:/Users/omar/Desktop/Code/AssureX/decision_engine.py)
  * [`config/thresholds.json`](file:///c:/Users/omar/Desktop/Code/AssureX/config/thresholds.json)
* **Changes Made by Me:**
  * Decoupled confidence gap boundaries from hardcoded values into an external configuration file `config/thresholds.json` so thresholds can be modified live during academic evaluation.
  * Added canonical class normalization (`normalize_class_label`) to reconcile differing label conventions between models.
  * Structured the decision explanation payload into `factors_supporting`, `factors_opposing`, `rules_passed`, `rules_failed`, and `evidence_needed`.
  * Added extensive docstrings and comments for evaluators to trace arbitration reasoning.
* **Testing I Performed:**
  * Created unit tests in `tests/test_decision_engine.py` verifying each consistency tier.
  * Tested zero-tolerance rule overrides: verified that even with 99% model confidence for Valid, a failed `EXCLUDED_DAMAGE_CHECK` or `WARRANTY_ACTIVE` rule forces a rejection or manual review.

---

### Entry 3: FastAPI Backend Architecture & Relational Domain Models
* **Tool Name:** Antigravity AI (Google DeepMind)
* **Purpose:** Build the core backend application providing JWT role-based authentication (Customer, Service Center, Claim Reviewer, Administrator), SQLAlchemy ORM entities (User, Product, Warranty, Claim, Document, RepairHistory, Prediction, Review, AuditLog), RESTful endpoints, SHA-256 duplicate detection, immutable audit logging, and database seeding.
* **Modules Affected:**
  * [`backend/main.py`](file:///c:/Users/omar/Desktop/Code/AssureX/backend/main.py)
  * [`backend/config.py`](file:///c:/Users/omar/Desktop/Code/AssureX/backend/config.py)
  * [`backend/database.py`](file:///c:/Users/omar/Desktop/Code/AssureX/backend/database.py)
  * [`backend/models.py`](file:///c:/Users/omar/Desktop/Code/AssureX/backend/models.py)
  * [`backend/auth.py`](file:///c:/Users/omar/Desktop/Code/AssureX/backend/auth.py)
  * [`backend/pipeline.py`](file:///c:/Users/omar/Desktop/Code/AssureX/backend/pipeline.py)
  * `backend/routes/*.py` (auth, products, warranties, claims, reviews, admin, repairs, ocr, export)
  * [`seed_db.py`](file:///c:/Users/omar/Desktop/Code/AssureX/seed_db.py)
  * [`requirements.txt`](file:///c:/Users/omar/Desktop/Code/AssureX/requirements.txt)
* **Changes Made by Me:**
  * Chose FastAPI with SQLAlchemy 2.0 for asynchronous request handling and OpenAPI interactive documentation.
  * Implemented SHA-256 hashing on document uploads to prevent receipt laundering across claims.
  * Created `seed_db.py` to seed default administrative accounts and populate 11 realistic evaluation claim scenarios.
* **Testing I Performed:**
  * Ran `python seed_db.py` to verify schema migrations and table creation.
  * Tested endpoint operations via Swagger UI at `http://localhost:8000/docs`.

---

### Entry 4: Receipt OCR Information Extraction Module
* **Tool Name:** Antigravity AI (Google DeepMind)
* **Purpose:** Implement `receipt_ocr.py` using EasyOCR and regex pattern matching to extract `purchase_date`, `invoice_number`, `product_name`, `model_number`, `serial_number`, `retailer`, and `purchase_amount` with field-level confidence flags.
* **Modules Affected:**
  * [`receipt_ocr.py`](file:///c:/Users/omar/Desktop/Code/AssureX/receipt_ocr.py)
  * `backend/routes/ocr.py`
* **Changes Made by Me:**
  * Added specialized regex patterns tailored for Pakistani tax documentation: 7-digit National Tax Numbers (NTN) and Sales Tax Registration Numbers (STRN).
  * Implemented confidence flags per extracted field so low-confidence extractions are highlighted for manual user verification.
  * Added fallback heuristics for thermal receipts with faded ink.
* **Testing I Performed:**
  * Executed unit tests in `tests/test_ocr_and_duplicates.py` using mock OCR outputs.
  * Tested extraction across diverse date formats (`YYYY-MM-DD`, `DD/MM/YYYY`, `DD-Mon-YYYY`).

---

### Entry 5: Responsive Web Frontend SPA
* **Tool Name:** Antigravity AI (Google DeepMind)
* **Purpose:** Construct a responsive, accessible single-page web frontend using Vanilla JavaScript, modern CSS, and Bootstrap 5.3, providing customer self-service portals, product registration, a 4-step claim submission wizard, claim tracker, reviewer queue, and admin analytics dashboard.
* **Modules Affected:**
  * [`frontend/index.html`](file:///c:/Users/omar/Desktop/Code/AssureX/frontend/index.html)
  * `frontend/css/style.css`
  * `frontend/js/app.js`
* **Changes Made by Me:**
  * Avoided heavy framework overhead by implementing lightweight Vanilla ES6+ modules with clean DOM rendering.
  * Integrated Chart.js to render live adjudication distribution bar/pie charts and volume trends.
  * Built an interactive 4-step wizard that presents real-time OCR extraction results before final submission.
  * Ensured responsive breakpoints for desktop, tablet, and mobile viewports.
* **Testing I Performed:**
  * Simulated user workflows across all 4 roles in Google Chrome and Microsoft Edge.
  * Tested viewport resizing from 375px mobile screens up to 1440px desktop displays.

---

### Entry 6: Pytest Test Suites & Demo Sample Claims Fixtures
* **Tool Name:** Antigravity AI (Google DeepMind)
* **Purpose:** Generate automated pytest test suites covering functional tests, integration tests, boundary conditions, negative tests, security/RBAC tests, and 11 mandatory demonstration claim fixtures in `sample_claims/`.
* **Modules Affected:**
  * `tests/conftest.py`
  * `tests/test_auth.py`
  * `tests/test_claims.py`
  * `tests/test_rule_engine.py`
  * `tests/test_decision_engine.py`
  * `tests/test_security_rbac.py`
  * `tests/test_ocr_and_duplicates.py`
  * `sample_claims/*.json` (11 demo fixture files)
* **Changes Made by Me:**
  * Configured `conftest.py` with an in-memory SQLite database and test client fixtures to prevent test contamination.
  * Created individual JSON fixtures in `sample_claims/` representing the required demo scenarios: valid claim, invalid claim, manual review, expired warranty, missing document, duplicate claim, contradictory date, serial mismatch, unauthorized repair, boundary date, and model disagreement.
* **Testing I Performed:**
  * Executed `python -m pytest` verifying all test suites execute cleanly with 100% pass rates.

---

### Entry 7: Security Audit & Removal of Auth Bypass Shortcuts
* **Tool Name:** Antigravity AI (Google DeepMind)
* **Purpose:** Conduct an audit to eliminate any 1-click evaluation shortcuts, demo-login bypass buttons, or hardcoded tokens, ensuring that all users authenticate strictly through normal password validation flows.
* **Modules Affected:**
  * [`backend/auth.py`](file:///c:/Users/omar/Desktop/Code/AssureX/backend/auth.py)
  * `backend/routes/auth.py`
  * `frontend/index.html`
  * `frontend/js/app.js`
  * [`CREDENTIALS.md`](file:///c:/Users/omar/Desktop/Code/AssureX/CREDENTIALS.md)
* **Changes Made by Me:**
  * Searched the entire codebase for mock login tokens and bypass logic and removed them completely.
  * Implemented dual-identifier login support (accepting either username or email with password).
  * Documented all seeded user accounts and plain-text passwords in `CREDENTIALS.md` for evaluator access.
* **Testing I Performed:**
  * Created `verify_all_roles_rbac.py` to confirm that unauthenticated and cross-role requests are blocked with HTTP 401/403.
  * Verified that all four roles can authenticate exclusively via valid credentials.

---

### Entry 8: Authentication Bug Root-Cause Diagnosis & Fix
* **Tool Name:** Antigravity AI (Google DeepMind)
* **Purpose:** Diagnose and resolve a blocking issue where seeded credentials triggered "Session Expired" errors on login.
* **Modules Affected:**
  * [`backend/auth.py`](file:///c:/Users/omar/Desktop/Code/AssureX/backend/auth.py)
  * `frontend/js/app.js`
  * [`test_login_end_to_end.py`](file:///c:/Users/omar/Desktop/Code/AssureX/test_login_end_to_end.py)
* **Changes Made by Me:**
  * Diagnosed root cause: timezone mismatch between UTC JWT token expiration calculations and local system clock, combined with cookie handling restrictions on `localhost`.
  * Added a 60-second decoding leeway to the JWT decoder to prevent premature expiration due to clock drift.
  * Standardized token transport using the `Authorization: Bearer <token>` HTTP header.
  * Wrote a dedicated automated validation script `test_login_end_to_end.py`.
* **Testing I Performed:**
  * Ran `python test_login_end_to_end.py` verifying successful login and profile retrieval for all 4 roles.

---

### Entry 9: ReportLab PDF Generation Engine Repair
* **Tool Name:** Antigravity AI (Google DeepMind)
* **Purpose:** Repair and optimize downloadable claim report PDF generation to produce official, downloadable adjudication certificates.
* **Modules Affected:**
  * `backend/routes/claims.py`
  * [`requirements.txt`](file:///c:/Users/omar/Desktop/Code/AssureX/requirements.txt)
* **Changes Made by Me:**
  * Eliminated fragile headless browser/HTML-to-PDF converters that caused system crashes on Windows.
  * Built a native ReportLab document generator creating structured flowables: branded header, product and claim details table, warranty timeline, rule engine audit checklist, and decision explanation.
* **Testing I Performed:**
  * Requested PDF downloads via the API (`/api/claims/{claim_id}/report`) and through the browser UI, confirming valid, styled PDF files are generated and downloaded.

---

### Entry 10: 30+ Unseen Test Claims Pipeline Execution & Comparison Report
* **Tool Name:** Antigravity AI (Google DeepMind)
* **Purpose:** Write and execute an automated evaluation script that runs 30+ unseen test claims through the full multimodal pipeline (Python model + GTM model + Rule Engine + Decision Engine) and outputs comprehensive CSV and Markdown comparison reports.
* **Modules Affected:**
  * [`run_model_comparison_pipeline.py`](file:///c:/Users/omar/Desktop/Code/AssureX/run_model_comparison_pipeline.py)
  * [`backend/pipeline.py`](file:///c:/Users/omar/Desktop/Code/AssureX/backend/pipeline.py)
  * [`reports/unseen_test_claims_report.csv`](file:///c:/Users/omar/Desktop/Code/AssureX/reports/unseen_test_claims_report.csv)
  * [`reports/unseen_test_claims_report.md`](file:///c:/Users/omar/Desktop/Code/AssureX/reports/unseen_test_claims_report.md)
* **Changes Made by Me:**
  * Mapped general electronics product categories (Audio, TV) to standard electronics policies instead of defaulting to mobile phone rules (which required smartphone IMEIs).
  * Clamped visual model probabilities to $[0.01, 1.00]$ and re-normalized them to sum strictly to $1.0$.
  * Fixed string comparison logic (`str.startswith("CORRECT")`) to prevent `"INCORRECT"` from matching as a substring.
  * Added dynamic one-line explanations for model disagreements, confidence gaps, and rule overrides.
* **Testing I Performed:**
  * Executed `python run_model_comparison_pipeline.py --num-claims 36`, confirming 36 claims evaluated, 88.9% model agreement, and 100.0% adjudication accuracy.
  * Verified all 14 requested columns are present and properly populated in both CSV and Markdown outputs.

---

### Entry 11: Comprehensive Project Report & Engineering Specification
* **Tool Name:** Antigravity AI (Google DeepMind)
* **Purpose:** Author a complete, academic-grade project report covering problem definition, background, architecture, module descriptions, database design, data dictionary, Mermaid diagrams, dataset generation, feature engineering, model training, actual empirical evaluation metrics, testing strategy, security, limitations, and future roadmap.
* **Modules Affected:**
  * [`documentation/PROJECT_REPORT.md`](file:///c:/Users/omar/Desktop/Code/AssureX/documentation/PROJECT_REPORT.md)
* **Changes Made by Me:**
  * Formatted complete architectural specifications into 5-tier system documentation.
  * Synthesized actual empirical metrics: Python Tabular Random Forest test accuracy of 92.0%, confusion matrix, and 88.9% multi-model pipeline agreement.
  * Provided complete, renderable Mermaid syntax for 5 required diagrams: Level 0/1 DFD, Use Case Diagram, Activity Diagram, Sequence Diagram, and Decision Flow Diagram.
* **Testing I Performed:**
  * Verified Mermaid diagram syntax rendering.
  * Audited cross-references against codebase files and database schemas.

---

## Summary of Verification & Testing Commands

To independently reproduce the testing and verification performed throughout the AI-assisted engineering process:

```powershell
# 1. Verify all unit, integration, security, and rule-engine tests
python -m pytest

# 2. Verify end-to-end credential authentication & session creation
python test_login_end_to_end.py

# 3. Verify Role-Based Access Control (RBAC) isolation
python verify_all_roles_rbac.py

# 4. Run the full 36 unseen claims evaluation pipeline
python run_model_comparison_pipeline.py

# 5. Launch the live application server
python run_server.py
```

---
*AssureX Engineering Team — AI Usage & Prompt Trajectory Log verified and approved.*
