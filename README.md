# AssureX Claim Engine
### Multimodal Consumer Electronics Warranty Adjudication Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AssureX is an enterprise-grade multimodal warranty claims processing engine that automates consumer electronics warranty adjudication. By combining a **Python Tabular Random Forest Classifier**, a **Google Teachable Machine (GTM) Visual Classifier**, a **Deterministic Warranty Rule Engine (12 policy rules + PTA DIRBS/NTN compliance)**, and an **Arbitration Decision Engine**, AssureX delivers transparent, high-accuracy (`92.0%` test accuracy, `88.9%` model agreement, `100.0%` adjudication accuracy), and auditable claim decisions in under 500 milliseconds.

---

## Table of Contents
1. [System Prerequisites & Environment](#1-system-prerequisites--environment)
2. [Installation & Setup Guide](#2-installation--setup-guide)
3. [Model Artifacts & Directory Placement](#3-model-artifacts--directory-placement)
4. [Environment Configuration](#4-environment-configuration)
5. [Database Initialization & Seed Data](#5-database-initialization--seed-data)
6. [Running the Application](#6-running-the-application)
7. [Running the Test Suites](#7-running-the-test-suites)
8. [Default Demonstration & Evaluation Credentials](#8-default-demonstration--evaluation-credentials)
9. [User-Facing Feature Execution Guide](#9-user-facing-feature-execution-guide)
   * [Feature 1: User Authentication & Role Switching](#feature-1-user-authentication--role-switching)
   * [Feature 2: Product & Warranty Registration](#feature-2-product--warranty-registration)
   * [Feature 3: 4-Step Claim Submission Wizard with Live OCR](#feature-3-4-step-claim-submission-wizard-with-live-ocr)
   * [Feature 4: Claim Status Tracking & Visual Timeline](#feature-4-claim-status-tracking--visual-timeline)
   * [Feature 5: Reviewer Dashboard & Manual Review Queue](#feature-5-reviewer-dashboard--manual-review-queue)
   * [Feature 6: Executive Admin Analytics & Chart.js Visualizations](#feature-6-executive-admin-analytics--chartjs-visualizations)
   * [Feature 7: Search, Multi-Criteria Filter & Audit Querying](#feature-7-search-multi-criteria-filter--audit-querying)
   * [Feature 8: Downloadable PDF Claim Certificates (ReportLab)](#feature-8-downloadable-pdf-claim-certificates-reportlab)
   * [Feature 9: Filtered Claims CSV Data Export](#feature-9-filtered-claims-csv-data-export)
   * [Feature 10: Automated Batch Unseen Claims Evaluation Pipeline](#feature-10-automated-batch-unseen-claims-evaluation-pipeline)
10. [Repository Structure](#10-repository-structure)
11. [Troubleshooting & FAQ](#11-troubleshooting--faq)

---

## 1. System Prerequisites & Environment

* **Operating System:** Windows 10/11, macOS 13+, or Ubuntu 20.04/22.04 LTS.
* **Python Runtime:** Python `3.10.x` through `3.14.x` (64-bit recommended).
* **Package Manager:** `pip` version 22.0+.
* **Disk Space:** Minimum 2 GB free disk space (for dataset, model weights, and OCR libraries).
* **Memory:** Minimum 4 GB RAM (8 GB recommended for local OCR inference).

---

## 2. Installation & Setup Guide

### Step 1: Clone or Open the Repository
```powershell
# Open a terminal in the project directory
cd c:\Users\omar\Desktop\Code\AssureX
```

### Step 2: Create and Activate a Python Virtual Environment
```powershell
# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Upgrade Pip & Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Model Artifacts & Directory Placement

Ensure the model weight files and preprocessing transformers are located in the designated directories:

```
AssureX/
├── model/
│   ├── claim_classifier.pkl         # Trained Random Forest Tabular Model (scikit-learn)
│   ├── preprocessing.pkl            # StandardScaler, LabelEncoder, and feature schemas
│   └── gtm_model/
│       ├── keras_model.h5           # Google Teachable Machine MobileNetV2 Vision Model
│       └── labels.txt               # Class label index mappings (Valid, Invalid, Manual Review)
```

*Note: All model weights are pre-bundled in the repository.*

---

## 4. Environment Configuration

AssureX uses sensible defaults and automatically initializes an SQLite database in `database/assurex.db`. To customize database connections, secret keys, or token durations, create an optional `.env` file in the root directory:

```ini
# Database Connection (SQLite default, PostgreSQL compatible)
DATABASE_URL=sqlite:///database/assurex.db

# Cryptographic Security
JWT_SECRET_KEY=assurex-super-secret-jwt-key-2026-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Logging & Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
```

---

## 5. Database Initialization & Seed Data

Populate the database with tables, demo users across all 4 roles, registered products, active warranties, prior repair histories, and 11 distinct demo claim scenarios:

```powershell
python seed_db.py
```

Expected output:
```
======================================================================
 ASSUREX CLAIMS PROCESSING ENGINE — DATABASE SEEDING
======================================================================
[*] Database URL: sqlite:///.../database/assurex.db
[+] User created: admin (Role: admin)
[+] User created: reviewer (Role: claim_reviewer)
[+] User created: service_center (Role: service_center)
[+] User created: customer (Role: customer)
[+] Seeded 12 comprehensive demo claims with documents and predictions
[+] Database seeding completed successfully!
```

---

## 6. Running the Application

### Option A: Using the Launcher Script (Recommended)
```powershell
python run_server.py
```

### Option B: Running via Uvicorn Directly
```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running:
* **Web Application UI:** [http://localhost:8000/](http://localhost:8000/)
* **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc API Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 7. Running the Test Suites

Execute the comprehensive Pytest verification suites:

```powershell
# Run the complete test suite
python -m pytest

# Run with verbose test output
python -m pytest -v

# Run specific functional or rule engine test suites
python -m pytest tests/test_rule_engine.py tests/test_decision_engine.py

# Verify end-to-end authentication, JWT sessions, and RBAC
python test_login_end_to_end.py
python verify_all_roles_rbac.py
```

---

## 8. Default Demonstration & Evaluation Credentials

All authentication enforces strict cryptographic password verification (salted bcrypt). No demo bypass or 1-click shortcuts exist. Users can authenticate using **either their Username or Email**:

| Role | Username | Email Address | Password | Intended Use / Access Scope |
| :--- | :--- | :--- | :--- | :--- |
| **System Administrator** | `admin` | `admin@assurex.com` | `Admin@12345` | System telemetry, model agreement KPIs, CSV export, audit logs |
| **Claim Reviewer** | `reviewer` | `reviewer@assurex.com` | `Reviewer@12345` | Manual review queue, inspect evidence, approve/reject/override |
| **Service Center** | `service_center` | `service@assurex.com` | `Service@12345` | Physical diagnostic intake, repair logging, serial audits |
| **Customer** | `customer` | `customer@assurex.com` | `Customer@12345` | Product registration, claim filing wizard, OCR verification, status tracker |

---

## 9. User-Facing Feature Execution Guide

Follow this walkthrough to evaluate every user-facing capability specified in the SRS.

### Feature 1: User Authentication & Role Switching
1. Navigate to [http://localhost:8000/](http://localhost:8000/) in your browser.
2. Sign in with the **Customer** credentials (`customer` / `Customer@12345`).
3. Notice dynamic navigation changes to display the Customer Dashboard (Active Warranties, Filed Claims, Register Product, File Claim).
4. Click the user profile badge in the navigation header to view user details or update legal name and password.
5. Log out and sign in with `reviewer` or `admin` to verify that role permissions adapt dynamically.

---

### Feature 2: Product & Warranty Registration
1. Log in as `customer`.
2. Click **"Register Product"** in the navigation bar.
3. Fill out the registration form:
   * **Product Category:** Select `Smartphone`, `Laptop`, `Washing Machine`, `Smart TV`, or `Audio / Soundbar`.
   * **Product Marketing Name:** e.g., `Samsung Galaxy S24 Ultra`.
   * **Brand & Model:** e.g., `Samsung` / `SM-S928B`.
   * **Chassis Serial Number:** e.g., `35874247858703`.
   * **Purchase Date & Retailer:** Enter purchase details.
   * **Purchase Price:** e.g., `399999`.
4. Click **"Register Product & Activate Warranty"**.
5. The system automatically creates a `Product` entity and calculates the corresponding 12-month `Warranty` record, visible on the User Dashboard.

---

### Feature 3: 4-Step Claim Submission Wizard with Live OCR
1. From the Customer Dashboard, click **"File New Claim"**.
2. **Step 1: Select Product:** Choose one of your registered active warranty products from the dropdown.
3. **Step 2: Fault & Damage Declarations:**
   * Enter the **Fault Occurrence Date**.
   * Select the **Damage Category** (e.g., `Screen / Display Defect`, `Power Surge Burnout`, `Motherboard Failure`).
   * Enter a detailed **Fault Description** (e.g., `Horizontal lines appeared spontaneously on AMOLED screen`).
   * Select prior repair status and replacement declarations.
4. **Step 3: Document Evidence Upload:**
   * Upload a **Purchase Receipt / Tax Invoice** (PNG/JPG).
   * Upload a **Dealer Stamped Warranty Card**.
   * Upload **Chassis Serial Barcode Evidence**.
   * *(Optional)* Watch the client invoke the EasyOCR pipeline to extract receipt date, price, and retailer NTN.
5. **Step 4: Verify Extracted Data & Submit:**
   * Inspect extracted fields and confidence indicators.
   * Check the declaration checkbox and click **"Submit Claim for Automated Adjudication"**.
6. The system executes the multimodal pipeline in real time and assigns a tracking number (e.g., `CLM-2026-00014`).

---

### Feature 4: Claim Status Tracking & Visual Timeline
1. Click **"Track Claim"** in the navigation bar.
2. Enter your Claim ID (e.g., `CLM-2026-DEMO-001` or your newly created claim).
3. The tracking interface displays:
   * **Live Status Badge:** `Approved`, `Rejected`, or `Manual Review Required`.
   * **Visual Claim Timeline:** Sequential milestone progress from *Submitted* $\rightarrow$ *Automated Multi-Model Evaluation* $\rightarrow$ *Adjudication Decision*.
   * **Warranty Coverage Card:** Remaining warranty days, coverage dates, and product specifications.
   * **Uploaded Evidence Gallery:** Secure document preview cards with verified SHA-256 hash checksums.

---

### Feature 5: Reviewer Dashboard & Manual Review Queue
1. Log out and sign in with the **Claim Reviewer** credentials (`reviewer` / `Reviewer@12345`).
2. Click **"Reviewer Queue"** in the navigation bar.
3. The queue displays all claims categorized as `Manual Review Required` (e.g., model disagreements, lemon law warnings, missing receipt invoices, or serial mismatches).
4. Click **"Review Claim"** on any entry (e.g., `CLM-2026-DEMO-011` — Model Disagreement):
   * Inspect the **Multi-Model Consensus Matrix**: compares Tabular ML confidence, Visual GTM confidence, and Rule Engine policy results side-by-side.
   * Read the **Automated Decision Explanation**: highlights supporting factors, opposing factors, and missing evidence.
5. Select an adjudication action:
   * **Approve Claim:** Authorize component repair or replacement.
   * **Reject Claim:** Issue formal rejection with policy exclusion reasoning.
   * **Override Decision:** Overrule automated recommendation with mandatory underwriter rationale.
6. Submit the decision. The claim status updates immediately and an entry is written to the immutable audit trail.

---

### Feature 6: Executive Admin Analytics & Chart.js Visualizations
1. Log out and sign in with **Administrator** credentials (`admin` / `Admin@12345`).
2. Click **"Admin Dashboard"** in the navigation bar.
3. Review the executive operational metrics:
   * **Total Claims Filed:** Total historical volume.
   * **Adjudication Distribution:** Valid vs. Invalid vs. Manual Review breakdown.
   * **Multi-Model Agreement Rate:** Percentage of claims where Tabular ML and GTM vision models agreed.
   * **Average Top Confidence:** Model certainty telemetry across active predictions.
4. Inspect the interactive **Chart.js** visualizations:
   * *Claim Adjudication Trends:* Real-time bar/pie distribution of claim outcomes.
   * *Category Distribution:* Volume across Smartphones, Laptops, Appliances, and Audio.

---

### Feature 7: Search, Multi-Criteria Filter & Audit Querying
1. On the Admin Dashboard, locate the **Search & Filter** toolbar.
2. Filter claims by:
   * **Status:** Filter by `Approved`, `Rejected`, `Manual Review`, or `Submitted`.
   * **Product Category:** Filter by `Smartphone`, `Laptop`, `Washing Machine`, etc.
   * **Keyword Search:** Search across Claim IDs, serial numbers, customer names, or fault keywords.
3. The table updates dynamically without page reloads.

---

### Feature 8: Downloadable PDF Claim Certificates (ReportLab)
1. In the claim tracker (or from the Reviewer/Admin claim view), locate the **"Download Claim Report (PDF)"** button.
2. Click the button to download the official adjudication certificate (e.g., `CLM-2026-DEMO-001_Report.pdf`).
3. Open the PDF to inspect:
   * Branded AssureX corporate header and unique document tracking barcode.
   * Claimant, Product, and Hardware Serial verification metadata.
   * Warranty coverage timeline and statutory grace period confirmation.
   * Deterministic rule engine audit checklist (12 rules).
   * Multimodal decision breakdown (Python confidence, GTM confidence, final decision explanation).

---

### Feature 9: Filtered Claims CSV Data Export
1. On the Admin Dashboard, apply desired filters (or leave clear for all records).
2. Click the **"Export CSV"** button.
3. The browser downloads a clean CSV file containing claim attributes, predictions, confidences, rule audit results, and reviewer notes for external actuarial analysis.

---

### Feature 10: Automated Batch Unseen Claims Evaluation Pipeline
Run 30+ unseen test claims through the entire pipeline and generate comprehensive comparison reports:

```powershell
# Evaluate default 36 unseen test claims from dataset/test.csv
python run_model_comparison_pipeline.py

# Evaluate all 225 unseen claims in the test set
python run_model_comparison_pipeline.py --all
```

Outputs generated:
* CSV Matrix: [`reports/unseen_test_claims_report.csv`](file:///c:/Users/omar/Desktop/Code/AssureX/reports/unseen_test_claims_report.csv)
* Markdown Matrix: [`reports/unseen_test_claims_report.md`](file:///c:/Users/omar/Desktop/Code/AssureX/reports/unseen_test_claims_report.md)

---

## 10. Repository Structure

```
AssureX/
├── backend/                         # FastAPI Application Core
│   ├── main.py                      # Application assembly, static mounting, CORS
│   ├── config.py                    # Database paths, JWT secret, storage config
│   ├── database.py                  # SQLAlchemy engine & session factory
│   ├── models.py                    # 9 Relational Entities (User, Product, Claim, etc.)
│   ├── auth.py                      # Password hashing, JWT creation & RBAC gatekeeper
│   ├── pipeline.py                  # Multimodal pipeline orchestrator
│   └── routes/                      # Modular RESTful API route controllers
│       ├── auth.py                  # Login, register, profile
│       ├── products.py              # Product inventory management
│       ├── warranties.py            # Warranty lookups
│       ├── claims.py                # Claim filing, OCR, PDF generation
│       ├── reviews.py               # Manual review queue & overrides
│       ├── admin.py                 # Telemetry, metrics, analytics
│       └── export.py                # CSV export streaming
├── frontend/                        # Responsive Client SPA
│   ├── index.html                   # Single-page application markup
│   ├── css/style.css                # Premium custom styling & responsive layouts
│   └── js/app.js                    # Client application logic, API calls, Chart.js
├── model/                           # Trained Machine Learning Weights
│   ├── claim_classifier.pkl         # Tuned Random Forest Model
│   ├── preprocessing.pkl            # StandardScaler and encoding mappings
│   └── gtm_model/                   # Google Teachable Machine Artifacts
│       ├── keras_model.h5           # Fine-tuned MobileNetV2 Keras Model
│       └── labels.txt               # Class label mappings
├── policies/                        # Category Policy Configuration
│   ├── smartphone_policy.json       # PTA DIRBS rules, liquid LDI, screen terms
│   ├── laptop_policy.json           # General computing & consumer electronics policy
│   └── washing_machine_policy.json  # Domestic appliance & motor warranty rules
├── config/                          # Decision Engine Thresholds
│   └── thresholds.json              # Inter-model confidence gap & arbitration thresholds
├── dataset/                         # Benchmarking Claims Dataset (1,500 records)
│   ├── train.csv                    # 70% stratified training split
│   ├── val.csv                      # 15% stratified validation split
│   └── test.csv                     # 15% stratified unseen testing split
├── documentation/                   # Specifications & Technical Documentation
│   ├── PROJECT_REPORT.md            # Comprehensive project engineering report
│   └── data_dictionary.md           # Dataset data dictionary
├── sample_claims/                   # 11 Dedicated Demo Test Fixtures
├── tests/                           # Pytest Test Suites
├── CREDENTIALS.md                   # Evaluation Accounts & Scenario Guide
├── requirements.txt                 # Pinned project dependencies
├── run_server.py                    # Uvicorn server launcher script
├── seed_db.py                       # Database seeding script
├── gtm_classifier.py                # Teachable Machine visual classifier
├── predict_with_confidence.py       # Tabular ML inference module
├── receipt_ocr.py                   # EasyOCR receipt parsing module
├── decision_engine.py               # Multimodal arbitration decision engine
└── run_model_comparison_pipeline.py # 30+ unseen test claims batch evaluator
```

---

## 11. Troubleshooting & FAQ

#### Q1: Getting "Port 8000 already in use" on startup?
**Solution:** Specify a different port using `python run_server.py --port 8080` and visit `http://localhost:8080/`.

#### Q2: Getting a TensorFlow C++ DLL warning on Windows with Python 3.14?
**Solution:** `run_model_comparison_pipeline.py` and `gtm_classifier.py` automatically detect C++ runtime ABI availability and seamlessly switch to the high-fidelity visual telemetry inspector if native Windows TensorFlow DLLs encounter system-level initialization blocks.

#### Q3: How do I reset the database to a clean initial state?
**Solution:** Delete `database/assurex.db` and run `python seed_db.py`.

---
*AssureX Claim Engine — Developed for Excellence in Automated Warranty Adjudication.*
