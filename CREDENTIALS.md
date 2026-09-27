# AssureX Claim Engine — Evaluation & Demo Credentials

This document provides credentials for the demonstration and evaluation accounts populated in the AssureX Claim Engine SQLite database via `seed_db.py`.

> **Note**: These accounts and credentials are created strictly for local development, academic defense, and evaluator testing.

---

## 1. Seeded User Accounts by Role

Users can sign in using **either their Username or Email Address** with their password:

| Role | Role Identifier | Display Name | Username | Email Address | Plaintext Password | Primary Access Scope |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Administrator** | `admin` | System Administrator | `admin` | `admin@assurex.com` | `Admin@12345` | System telemetry, model agreement KPIs, CSV/Excel exports, policy configs, audit logs |
| **Claim Reviewer** | `claim_reviewer` | Senior Claims Assessor | `reviewer` | `reviewer@assurex.com` | `Reviewer@12345` | Manual review queue, underwriter notes, decision overrides, claim inspection |
| **Service Center** | `service_center` | TechnoCity Authorized Hub | `service_center` | `service@assurex.com` | `Service@12345` | Physical diagnostic intake, repair history recording, technical fault reporting |
| **Customer** | `customer` | Ali Khan | `customer` | `customer@assurex.com` | `Customer@12345` | Product registration, claim submission wizard, OCR receipt verification, status tracker |

---

## 2. Authentication & Session Architecture

All authentication flows through standard OAuth2 Password Bearer flow via `/api/auth/login` and `/api/auth/token` with secure salted PBKDF2-HMAC-SHA256 password hashing.

- **Dual Identifier Support**: Users may authenticate using either their `username` (e.g. `admin`) or their `email` (e.g. `admin@assurex.com`).
- **No Shortcuts / No Bypass**: Hardcoded 1-click evaluation shortcuts, demo login buttons, and bypass routes have been **completely eliminated**.
- **Role-Based Access Control (RBAC)**: Enforced dynamically on every endpoint via FastAPI dependency injection (`require_roles`).
- **Profile Updates**: Users can view and update their profile details (Full Name, Email, Password) securely from the navigation bar.
- **Clock Skew Resilience**: JWT tokens utilize timezone-aware UTC timestamps with a 60-second decoding leeway to prevent premature "Session Expired" states.

---

## 3. Seeded Claims & Demo Scenarios Guide

The database is seeded with 11 demo scenarios (plus 1 baseline duplicate reference) located in `sample_claims/`:

| Scenario ID | Demo Scenario | Claim ID | Product / Serial | Expected Status | Core Verification / Trigger Tested |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Reference Duplicate Base | `CLM-2026-00042` | HP Pavilion (`HP-PAV-15-77291-KHI`) | `Approved` | Reference claim storing original invoice SHA-256 hash |
| **Demo 1** | Valid Claim | `CLM-2026-DEMO-001` | Samsung S24 Ultra (`35874247858703`) | `Approved` | Consensus Strong Match, valid receipt, active warranty |
| **Demo 2** | Invalid Claim | `CLM-2026-DEMO-002` | Apple MacBook Pro (`C02G8712MD6T`) | `Rejected` | Excluded damage policy violation (severe liquid spill) |
| **Demo 3** | Manual Review Claim | `CLM-2026-DEMO-003` | Haier Inverter AC (`HAIER-INV-992144`) | `Manual Review` | Lemon Law trigger (3 consecutive motor stator failures) |
| **Demo 4** | Expired Warranty Claim | `CLM-2026-DEMO-004` | Dell XPS 15 (`DELL-XPS-48192-KHI`) | `Rejected` | Calendar-month warranty expired 365 days prior to filing |
| **Demo 5** | Missing Document Claim | `CLM-2026-DEMO-005` | Sony Bravia OLED (`SONY-OLED-65-8812`) | `Manual Review` | Missing mandatory proof of purchase receipt/invoice |
| **Demo 6** | Duplicate Claim | `CLM-2026-DEMO-006` | HP Pavilion (`HP-PAV-15-77291-KHI`) | `Rejected` | SHA-256 hash collision against `CLM-2026-00042` |
| **Demo 7** | Contradictory Claim | `CLM-2026-DEMO-007` | LG Washing Machine (`LG-WM-88219-LHR`) | `Manual Review` | Temporal paradox (claim filing date precedes purchase date) |
| **Demo 8** | Serial Mismatch Claim | `CLM-2026-DEMO-008` | iPad Pro 12.9 (`DLXN12349018`) | `Manual Review` | OCR receipt serial does not match physical chassis serial |
| **Demo 9** | Unauthorized Repair Claim | `CLM-2026-DEMO-009` | Canon EOS R6 (`CN-EOS-R6-551928`) | `Rejected` | Unauthorized third-party technician modification / broken seals |
| **Demo 10** | Boundary Date Claim | `CLM-2026-DEMO-010` | Asus ROG Zephyrus (`ROG-ZEPH-88120-ISB`) | `Approved` | Claim filed on the exact calendar day of warranty expiration |
| **Demo 11** | Model Disagreement Claim | `CLM-2026-DEMO-011` | PlayStation 5 (`PS5-CFI-1216A-99`) | `Manual Review` | Tabular ML (Likely Valid) vs Visual GTM (Likely Invalid) |

---

## 4. Verification and Re-seeding Commands

To verify all credentials and session creation end-to-end:

```powershell
python test_login_end_to_end.py
```

To cleanly wipe and re-populate the demo database:

```powershell
python seed_db.py
```
