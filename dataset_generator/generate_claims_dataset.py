"""
Warranty Claims Synthetic Dataset Generator - AssureX Machine Learning Module
=============================================================================

This script generates a high-fidelity synthetic dataset of 1,500 consumer warranty
claim records for multi-class claims classification:
  1. Valid Claim   (500 records)
  2. Invalid Claim (500 records)
  3. Manual Review (500 records)

Features & Dimensions (26 fields):
----------------------------------
- claim_id                    : Unique alphanumeric identifier (CLM-XXXX-YYYYY)
- user_id                     : Customer identifier (USR-XXXXX)
- product_id                  : SKU / Catalog identifier (PRD-CAT-XXXX)
- product_category            : Consumer electronics category (Smartphone, Laptop, Washing Machine, Smart TV, Audio)
- product_name                : Full commercial product title
- brand                       : Manufacturer / OEM Brand
- model_number                : Specific equipment model code
- serial_number               : Unique chassis serial number or IMEI
- purchase_date               : Date of retail purchase (YYYY-MM-DD)
- purchase_price              : Invoiced purchase amount
- retailer                    : Invoicing dealer or national distributor outlet
- warranty_duration_months    : Policy coverage duration (12, 24, 36 months)
- warranty_start_date         : Inception date of warranty coverage (YYYY-MM-DD)
- warranty_expiry_date        : Expiration boundary date (YYYY-MM-DD)
- fault_occurrence_date       : Date the hardware malfunction was first noticed (YYYY-MM-DD)
- fault_description           : Customer-provided textual description of symptoms/defects
- damage_type                 : Categorical classification of defect / damage cause
- claim_submission_date       : Formal claim filing date with customer service (YYYY-MM-DD)
- repair_history              : Number of past service interventions and authorization flags
- has_receipt                 : Boolean flag indicating presence of retail tax invoice
- has_warranty_card           : Boolean flag indicating presence of stamped warranty card
- has_product_image           : Boolean flag indicating presence of photo evidence
- has_serial_evidence         : Boolean flag indicating presence of barcode/nameplate photo
- serial_number_on_receipt    : Serial number printed on purchase receipt (checks for mismatch)
- prior_replacement           : Boolean flag indicating whether unit has already been replaced
- class_label                 : Target classification ("Valid Claim", "Invalid Claim", "Manual Review")

================================================================================
EXPLICIT GENERATION LOGIC SPECIFICATION:
================================================================================

1. VALID CLAIM (Target: 500 records):
   - Temporal Validity: Fault occurrence is strictly within the active warranty window
     (after warranty_start_date and comfortably before warranty_expiry_date).
   - Fault Coverage: Damage type represents bona fide manufacturing or hardware failure
     (e.g., Motherboard Circuit Failure, Inverter Motor Burnout, AMOLED Green Line Defect,
     Display Backlight Failure, Battery Health Degradation under threshold).
   - Document Integrity: All documentation present (has_receipt=True, has_warranty_card=True,
     has_product_image=True, has_serial_evidence=True).
   - Identity Verification: Receipt serial matches physical equipment serial exactly.
   - Timeliness: Claim submission is filed within the standard reporting period (1-7 days).
   - Integrity & History: Clean repair history (0 prior repairs or 1-2 strictly authorized
     service center repairs); prior_replacement is False; zero chronological contradictions.

2. INVALID CLAIM (Target: 500 records):
   Explicit grounds for automated rejection (at least one or a combination):
   - Rule Breach A (Warranty Expired): Fault occurred well after warranty expiry (15 to 365+ days late).
   - Rule Breach B (Excluded Damage): Defect caused by uninsurable external abuse:
     * Liquid / moisture ingress (triggered LCIs/LDIs).
     * Physical drop / impact fracture / cracked glass.
     * Electrical power surge burnout without mandatory voltage stabilizer.
     * Pest / rodent infestation inside appliance cabinet.
     * Customer software tampering (unlocked bootloader, custom ROM, rooted kernel).
     * Normal cosmetic wear and tear.
   - Rule Breach C (No Proof of Purchase): Missing retail invoice (has_receipt=False),
     resulting in missing receipt serial.
   - Rule Breach D (Serial Mismatch): Serial number on receipt does not match unit serial
     (warranty laundering or swapped receipt).
   - Rule Breach E (Fraud / Double Replacement): prior_replacement=True (device was already
     replaced previously; original serial is void).
   - Rule Breach F (Gross Late Reporting): Claim filed months after fault without justification.

3. MANUAL REVIEW (Target: 500 records):
   Borderline / ambiguous cases requiring human claims adjuster discretion:
   - Scenario A (Partial Documentation): 1-2 missing supplementary documents (e.g., receipt
     is present, but warranty card or serial evidence photo is missing/blurry).
   - Scenario B (Boundary / Grace Dates):
     * Fault occurred within 1-10 days of warranty expiration date.
     * Claim submitted during the statutory grace period (1-15 days post-expiry).
     * Dead-on-Arrival (DOA) borderline cases occurring immediately post-purchase.
   - Scenario C (Plausible / Mixed Repair History):
     * 1 unauthorized third-party repair declared for a non-critical accessory/module.
     * Repeat repair threshold reached (3 prior authorized repairs, triggering Lemon review).
   - Scenario D (Minor Discrepancies & Transcription Errors):
     * Single-character OCR / handwritten receipt typo on serial number (e.g., '0' vs 'O', '8' vs 'B').
     * Slight reporting delay (e.g., 8-15 days post-fault, exceeding 7-day guidance due to holidays).

================================================================================
CONTROLLED NOISE INJECTION (~10% across dataset):
================================================================================
To prevent machine learning models from memorizing rigid deterministic rules or
overfitting to perfectly synthetic patterns, ~10% of records in each class receive
realistic real-world messy nuances:
- Valid Claims with noise: Clerical date typo (e.g. claim date entered 1 day before fault
  due to time-zone / system logging error), single-character receipt transcription typo,
  or missing non-critical product photo (in-store technician verified physically).
- Invalid Claims with noise: Masked defect text (damage_type marked as "Hardware Glitch"
  while text admits dropped in water), or all document checkboxes marked True despite
  an expired warranty.
- Manual Review with noise: Incomplete retailer metadata, NaN / missing optional fields,
  or unusual promotional purchase prices.

Stratified Splitting:
---------------------
Stratified 70/15/15 split (train: 1,050, val: 225, test: 225) preserving exact 1:1:1
class balance across all subsets. Saves train.csv, val.csv, test.csv, and dataset_stats.json.
"""

import os
import json
import random
import argparse
from datetime import date, timedelta
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
from faker import Faker

# Initialize Faker with seed for consistent determinism
SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# ==============================================================================
# CATALOG & DOMAIN REFERENCE DATA (Tailored to AssureX Electronics Policies)
# ==============================================================================

CATEGORIES_CONFIG = {
    "Smartphone": {
        "brands": {
            "Samsung": [
                ("Galaxy S24 Ultra", "SM-S928B", (180000, 360000)),
                ("Galaxy A55 5G", "SM-A556E", (75000, 130000)),
                ("Galaxy Z Fold 5", "SM-F946B", (320000, 490000)),
            ],
            "Apple": [
                ("iPhone 15 Pro Max", "A3106", (320000, 480000)),
                ("iPhone 15", "A3090", (220000, 310000)),
                ("iPhone 14", "A2882", (180000, 240000)),
            ],
            "Xiaomi": [
                ("Xiaomi 14 Ultra", "24030PN60G", (190000, 280000)),
                ("Redmi Note 13 Pro+", "23090RA98G", (65000, 110000)),
            ],
        },
        "covered_faults": [
            ("Logic board PMIC power distribution failure; unit will not boot", "Logic Board Failure"),
            ("Spontaneous vertical green line on AMOLED panel post official OTA update", "Display Panel Defect"),
            ("Camera module Optical Image Stabilization (OIS) actuator rattle and blur", "Camera Module Hardware Defect"),
            ("USB-C fast charging handshake pin solder fatigue; fails to negotiate charge", "Charging Port Failure"),
            ("Cellular baseband modem chip malfunction; persistent No Service state", "Modem / Cellular Hardware Defect"),
            ("Earpiece acoustic transducer failure causing distorted voice audio", "Acoustic Transducer Defect"),
        ],
        "excluded_faults": [
            ("Handset dropped into swimming pool; internal SIM tray LDI triggered vivid red", "Liquid / Moisture Damage"),
            ("Front display glass shattered with circular impact point from concrete drop", "Physical Impact / Accidental Drop"),
            ("Device bootloader unlocked and custom kernel flashed; Knox counter tripped 0x1", "Unauthorized Modification / Rooting"),
            ("Third-party replacement LCD installed at local repair market; missing thermal bracket", "Unauthorized Tampering / Bazaar Rework"),
            ("Minor hairline scratches on aluminum frame from pocket grit", "Normal Cosmetic Wear"),
        ],
        "durations": [12],
    },
    "Laptop": {
        "brands": {
            "Lenovo": [
                ("ThinkPad T14 Gen 4", "21HD001EUS", (190000, 340000)),
                ("IdeaPad Slim 5", "82XD0006US", (110000, 185000)),
                ("Legion Pro 5i", "82WK0046US", (260000, 420000)),
            ],
            "Dell": [
                ("XPS 15 9530", "XPS9530-7456SLV", (280000, 460000)),
                ("Latitude 5440", "L5440-CTO1", (160000, 260000)),
                ("Inspiron 16", "i5630-7212SLV", (120000, 195000)),
            ],
            "HP": [
                ("Spectre x360 14", "14-ef2013dx", (250000, 390000)),
                ("EliteBook 840 G10", "818M8UT", (195000, 310000)),
                ("Pavilion 15", "15-eg3023nr", (95000, 160000)),
            ],
            "Apple": [
                ("MacBook Pro 14 M3", "A2992", (310000, 490000)),
                ("MacBook Air 13 M2", "A2681", (190000, 280000)),
            ],
        },
        "covered_faults": [
            ("Motherboard MOSFET short circuit causing instantaneous thermal protection shutdown", "Motherboard Circuit Failure"),
            ("Dedicated GPU VRAM solder fatigue resulting in visual polygon artifacting", "GPU Hardware Malfunction"),
            ("Factory NVMe SSD controller locked in permanent read-only state", "Internal Storage Degradation"),
            ("Internal cooling fan bearing seizure and zero RPM tachometer error", "Cooling System Defect"),
            ("Keyboard matrix row 3 (keys A-S-D-F) non-responsive under dry conditions", "Keyboard Matrix Failure"),
            ("Persistent dead pixel vertical line across IPS display panel", "Display Panel Defect"),
        ],
        "excluded_faults": [
            ("Tea spill across keyboard assembly; red liquid contact indicators observed on motherboard", "Liquid / Moisture Damage"),
            ("Laptop crushed inside backpack; cracked matrix glass and bent aluminum chassis", "Physical Impact / Accidental Drop"),
            ("Motherboard charging rail burned out using ungrounded third-party generic AC adapter", "Power Surge Burnout (No Voltage Stabilizer)"),
            ("Unauthorized BGA chip reflow and jumper wire bypass attempted at local computer plaza", "Unauthorized Tampering / Bazaar Rework"),
            ("Operating system locked with forgotten BitLocker recovery key and BIOS supervisor password", "Customer Software / Configuration Issue"),
        ],
        "durations": [12, 24],
    },
    "Washing Machine": {
        "brands": {
            "Haier": [
                ("Inverter Front-Load 10kg Washer", "HWM100-1678", (85000, 160000)),
                ("Direct Drive Top-Load 12kg Washer", "HWM120-826", (65000, 110000)),
                ("Twin Tub Semi-Automatic Washer", "HWM80-50", (32000, 52000)),
            ],
            "Dawlance": [
                ("Inverter Front-Load ProWash", "DW-FL-9000", (82000, 145000)),
                ("Top-Load Energy Saver 11kg", "DW-9060-ES", (58000, 95000)),
            ],
            "Samsung": [
                ("EcoBubble Front-Load Inverter 9kg", "WW90T554DAW", (110000, 195000)),
                ("Wobble Technology Top-Load 13kg", "WA13CG5441BY", (78000, 130000)),
            ],
            "LG": [
                ("AI DD Front-Load Washer 9kg", "F4V5VYP0W", (125000, 215000)),
                ("Smart Inverter Top-Load 11kg", "T2111VSSV", (72000, 120000)),
            ],
        },
        "covered_faults": [
            ("Inverter Direct Drive motor stator hall-effect sensor electrical open circuit", "Inverter Motor Failure"),
            ("Main electronic drive controller PCB failed with IGBT gate driver error E2", "Main Control Board Defect"),
            ("Water inlet dual solenoid valve coil burnout; machine fails to fill water", "Water Valve Failure"),
            ("Internal drainage pump motor winding electrical open circuit with OE code", "Drain Pump Defect"),
            ("Electronic door safety microswitch interlock latch contact failure", "Safety Interlock Defect"),
            ("Drum balance suspension damper mechanical collapse under rated wash load", "Mechanical Suspension Defect"),
        ],
        "excluded_faults": [
            ("High-voltage lightning surge burned PCB while plugged directly into AC mains without 15A stabilizer", "Power Surge Burnout (No Voltage Stabilizer)"),
            ("Electrical wiring harness chewed through and PCB nested by rodents inside cabinet base", "Pest / Rodent Infestation"),
            ("Excessive limescale encrustation and tub corrosion from brackish 2200 PPM groundwater", "High-TDS Groundwater Corrosion"),
            ("Machine used continuously in a commercial hostel laundromat (exceeding 8 cycles/day)", "Commercial Duty Overuse"),
            ("Coin and hairpin stuck in drain pump propeller fracturing the impeller housing", "Foreign Object Physical Obstruction"),
            ("Local bazaar technician attempted motor rewinding with non-spec copper wire", "Unauthorized Tampering / Bazaar Rework"),
        ],
        "durations": [12, 24, 36],
    },
    "Smart TV": {
        "brands": {
            "Sony": [
                ("Bravia 4K OLED 55-inch", "XR-55A80L", (280000, 480000)),
                ("Bravia LED 4K 65-inch", "KD-65X77L", (180000, 290000)),
            ],
            "Samsung": [
                ("QLED 4K Smart TV 55-inch", "QA55Q60CA", (160000, 270000)),
                ("Crystal UHD 4K 65-inch", "UA65CU7000", (140000, 230000)),
            ],
            "TCL": [
                ("QLED Mini-LED 55-inch", "55C755", (110000, 185000)),
                ("4K HDR Smart TV 50-inch", "50P635", (65000, 105000)),
            ],
        },
        "covered_faults": [
            ("Internal T-CON timing controller logic failure causing display split-screen inversion", "Timing Controller (T-CON) Defect"),
            ("LED backlight string driver open circuit resulting in total black screen with intact audio", "Backlight Array Failure"),
            ("Internal switch-mode power supply (SMPS) secondary rail diode breakdown", "Internal Power Supply Failure"),
            ("Motherboard HDMI receiver chip failure; all ports report No Signal", "Mainboard HDMI Interface Failure"),
        ],
        "excluded_faults": [
            ("Direct impact cracked display panel; internal LCD matrix ruptured with black spidering", "Physical Impact / Accidental Drop"),
            ("Severe lightning strike destroyed primary power circuitry during thunderstorm", "Power Surge Burnout (No Voltage Stabilizer)"),
            ("Liquid spray applied directly to glass panel corroded lower COF ribbon flex cables", "Liquid / Moisture Damage"),
        ],
        "durations": [12, 24],
    },
    "Audio / Soundbar": {
        "brands": {
            "Sony": [
                ("Dolby Atmos Soundbar 5.1", "HT-S40R", (65000, 115000)),
                ("Wireless Noise Canceling Headphones", "WH-1000XM5", (75000, 120000)),
            ],
            "Samsung": [
                ("Q-Series Soundbar 3.1.2ch", "HW-Q600C", (70000, 125000)),
            ],
            "Apple": [
                ("AirPods Pro 2nd Gen", "MTJV3", (55000, 85000)),
            ],
        },
        "covered_faults": [
            ("Bluetooth RF transceiver chip desoldered; device refuses pairing handshakes", "Wireless Transceiver Defect"),
            ("Active subwoofer Class-D amplifier module silent; power LED steady green", "Amplifier Circuit Failure"),
            ("ANC feedback microphone capsule dead; severe acoustic howling artifact", "Microphone Hardware Failure"),
        ],
        "excluded_faults": [
            ("Left earpiece dropped in water puddle; acoustic chamber liquid contaminated", "Liquid / Moisture Damage"),
            ("Headband hinge snapped in half from severe outward hyperextension", "Physical Impact / Accidental Drop"),
            ("Subwoofer power coil fried by 280V voltage spike on unprotected extension lead", "Power Surge Burnout (No Voltage Stabilizer)"),
        ],
        "durations": [12],
    },
}

AUTHORIZED_RETAILERS = [
    "Airlink Communications Official Flagship",
    "Muller & Phipps Authorized Retail",
    "Mega PK Technologies Flagship Hub",
    "Metro Cash & Carry Official Counter",
    "Daraz Mall Authorized Brand Store",
    "Cnergy Tech Retail Center",
    "Hafeez Center Certified Dealer Outlets",
    "TechnoCity Prime Electronics",
    "Al-Madina Electronics Authorized Franchise",
    "Brand Flagship Experience Store",
]

# ==============================================================================
# HELPER GENERATORS
# ==============================================================================

def generate_serial_number(category: str, brand: str) -> str:
    """Generate realistic manufacturer serial numbers / IMEIs."""
    cat_code = category[:2].upper()
    brand_code = brand[:3].upper()
    if category == "Smartphone":
        # Realistic 15-digit TAC + Serial IMEI format
        prefix = random.choice(["860492", "358742", "864109", "359124"])
        body = f"{random.randint(10000000, 99999999):08d}"
        luhn = str(random.randint(0, 9))
        return f"{prefix}{body[:7]}{luhn}"
    else:
        chars = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"
        random_suffix = "".join(random.choices(chars, k=7))
        return f"{brand_code}-{cat_code}-2025-{random_suffix}"


def apply_minor_transcription_typo(serial: str) -> str:
    """Apply realistic optical/clerical character substitution."""
    substitutions = {
        "0": "O", "O": "0",
        "1": "I", "I": "1",
        "8": "B", "B": "8",
        "5": "S", "S": "5",
        "2": "Z", "Z": "2",
    }
    serial_chars = list(serial)
    for idx, ch in enumerate(serial_chars):
        if ch in substitutions:
            serial_chars[idx] = substitutions[ch]
            return "".join(serial_chars)
    # If no substitute found, change the last character
    if serial_chars:
        serial_chars[-1] = "X" if serial_chars[-1] != "X" else "Y"
    return "".join(serial_chars)


def generate_product_info() -> Tuple[str, str, str, str, str, float, int]:
    """Randomly pick a realistic product from the catalog."""
    category = random.choice(list(CATEGORIES_CONFIG.keys()))
    cat_data = CATEGORIES_CONFIG[category]
    brand = random.choice(list(cat_data["brands"].keys()))
    model_name, model_number, price_range = random.choice(cat_data["brands"][brand])
    price = round(random.uniform(price_range[0], price_range[1]), -2)
    warranty_months = random.choice(cat_data["durations"])
    product_id = f"PRD-{category[:2].upper()}-{brand[:3].upper()}-{model_number[:4]}"
    return category, brand, model_name, model_number, product_id, price, warranty_months


# ==============================================================================
# CLASS-SPECIFIC RECORD BUILDERS
# ==============================================================================

def generate_valid_claim(claim_idx: int) -> Dict[str, Any]:
    """
    GENERATION LOGIC FOR VALID CLAIMS (Standard Profile):
    - Temporal: Fault occurred well within active warranty window.
    - Fault: Covered hardware manufacturing defect from policy catalog.
    - Documents: All 4 documents present (receipt, card, image, serial photo).
    - Serial Match: Receipt serial matches physical equipment serial.
    - Reporting: Claim filed 1-5 days post-fault (well within 7-day limit).
    - Repair History: Clean (0 prior repairs or 1-2 strictly authorized).
    - Prior Replacement: False.
    """
    category, brand, model_name, model_number, product_id, price, warranty_months = generate_product_info()
    serial = generate_serial_number(category, brand)
    receipt_serial = serial  # Perfect match

    # Temporal sequencing
    # Base anchor date: purchase between 30 and 300 days ago
    purchase_days_ago = random.randint(45, min(365, warranty_months * 30 - 30))
    purchase_dt = date.today() - timedelta(days=purchase_days_ago)
    warranty_start_dt = purchase_dt + timedelta(days=random.choice([0, 1]))
    warranty_expiry_dt = warranty_start_dt + timedelta(days=int(warranty_months * 30.4375))

    # Fault occurs while warranty is actively valid (comfortably before expiry)
    days_active = max(10, (warranty_expiry_dt - warranty_start_dt).days - 30)
    fault_elapsed = random.randint(10, days_active)
    fault_dt = warranty_start_dt + timedelta(days=fault_elapsed)

    # Claim filed promptly (within 1 to 5 days of fault)
    reporting_delay = random.randint(1, 5)
    claim_dt = fault_dt + timedelta(days=reporting_delay)

    # Covered fault selection
    fault_desc, damage_type = random.choice(CATEGORIES_CONFIG[category]["covered_faults"])

    # Authorized repair history
    repair_options = [
        "0 repairs",
        "0 repairs",
        "0 repairs",
        "1 repair (Authorized Service Center)",
        "2 repairs (Authorized Service Center)",
    ]
    repair_hist = random.choice(repair_options)

    return {
        "claim_id": f"CLM-2026-{claim_idx:05d}",
        "user_id": f"USR-{random.randint(10000, 99999)}",
        "product_id": product_id,
        "product_category": category,
        "product_name": f"{brand} {model_name}",
        "brand": brand,
        "model_number": model_number,
        "serial_number": serial,
        "purchase_date": purchase_dt.isoformat(),
        "purchase_price": price,
        "retailer": random.choice(AUTHORIZED_RETAILERS),
        "warranty_duration_months": warranty_months,
        "warranty_start_date": warranty_start_dt.isoformat(),
        "warranty_expiry_date": warranty_expiry_dt.isoformat(),
        "fault_occurrence_date": fault_dt.isoformat(),
        "fault_description": fault_desc,
        "damage_type": damage_type,
        "claim_submission_date": claim_dt.isoformat(),
        "repair_history": repair_hist,
        "has_receipt": True,
        "has_warranty_card": True,
        "has_product_image": True,
        "has_serial_evidence": True,
        "serial_number_on_receipt": receipt_serial,
        "prior_replacement": False,
        "class_label": "Valid Claim",
    }


def generate_invalid_claim(claim_idx: int) -> Dict[str, Any]:
    """
    GENERATION LOGIC FOR INVALID CLAIMS (Definitive Rejection Grounds):
    At least one (or a combination) of explicit disqualifiers:
    - Reason A: Warranty expired (fault occurred 20-300 days after expiration).
    - Reason B: Excluded damage (liquid, drop/shatter, surge without stabilizer, rodents).
    - Reason C: No proof of purchase (has_receipt=False, receipt serial is missing).
    - Reason D: Serial mismatch (different device serial number on receipt).
    - Reason E: Prior replacement already claimed (prior_replacement=True).
    - Reason F: Gross late reporting (filed 45-120 days post-incident).
    """
    category, brand, model_name, model_number, product_id, price, warranty_months = generate_product_info()
    serial = generate_serial_number(category, brand)

    # Pick primary failure mode for invalidation
    failure_mode = random.choice([
        "EXPIRED_WARRANTY",
        "EXCLUDED_DAMAGE",
        "NO_RECEIPT",
        "SERIAL_MISMATCH",
        "PRIOR_REPLACEMENT_EXHAUSTED",
        "GROSS_REPORTING_DELAY",
    ])

    # Base dates
    purchase_days_ago = random.randint(200, 800)
    purchase_dt = date.today() - timedelta(days=purchase_days_ago)
    warranty_start_dt = purchase_dt + timedelta(days=1)
    warranty_expiry_dt = warranty_start_dt + timedelta(days=int(warranty_months * 30.4375))

    # Defaults
    has_receipt = True
    has_warranty_card = True
    has_product_image = True
    has_serial_evidence = True
    receipt_serial = serial
    prior_replacement = False
    repair_hist = random.choice([
        "0 repairs",
        "1 repair (Authorized Service Center)",
        "1 repair (Unauthorized Third-Party)",
    ])

    if failure_mode == "EXPIRED_WARRANTY":
        # Fault happened well AFTER warranty expiry date
        days_after_expiry = random.randint(25, 240)
        fault_dt = warranty_expiry_dt + timedelta(days=days_after_expiry)
        claim_dt = fault_dt + timedelta(days=random.randint(1, 7))
        fault_desc, damage_type = random.choice(CATEGORIES_CONFIG[category]["covered_faults"])

    elif failure_mode == "EXCLUDED_DAMAGE":
        # Fault is an explicit exclusion (water, shatter, surge, tamper)
        fault_dt = warranty_start_dt + timedelta(days=random.randint(15, 150))
        claim_dt = fault_dt + timedelta(days=random.randint(1, 5))
        fault_desc, damage_type = random.choice(CATEGORIES_CONFIG[category]["excluded_faults"])

    elif failure_mode == "NO_RECEIPT":
        # No proof of purchase
        has_receipt = False
        receipt_serial = None  # No receipt available
        has_warranty_card = random.choice([True, False])
        fault_dt = warranty_start_dt + timedelta(days=random.randint(20, 180))
        claim_dt = fault_dt + timedelta(days=random.randint(1, 6))
        fault_desc, damage_type = random.choice(CATEGORIES_CONFIG[category]["covered_faults"])

    elif failure_mode == "SERIAL_MISMATCH":
        # Serial on receipt belongs to a different device
        other_serial = generate_serial_number(category, brand)
        receipt_serial = other_serial
        fault_dt = warranty_start_dt + timedelta(days=random.randint(20, 180))
        claim_dt = fault_dt + timedelta(days=random.randint(1, 6))
        fault_desc, damage_type = random.choice(CATEGORIES_CONFIG[category]["covered_faults"])

    elif failure_mode == "PRIOR_REPLACEMENT_EXHAUSTED":
        # Device was already replaced under warranty; claiming on void contract
        prior_replacement = True
        fault_dt = warranty_start_dt + timedelta(days=random.randint(20, 180))
        claim_dt = fault_dt + timedelta(days=random.randint(1, 6))
        fault_desc = "Unit power failure; replacement unit already issued under previous claim"
        damage_type = "Hardware Failure"

    else:  # GROSS_REPORTING_DELAY
        # Fault occurred months ago and customer neglected to file within 7-day reporting period
        fault_dt = warranty_start_dt + timedelta(days=random.randint(20, 100))
        claim_dt = fault_dt + timedelta(days=random.randint(45, 120))  # Massively delayed
        fault_desc, damage_type = random.choice(CATEGORIES_CONFIG[category]["covered_faults"])

    return {
        "claim_id": f"CLM-2026-{claim_idx:05d}",
        "user_id": f"USR-{random.randint(10000, 99999)}",
        "product_id": product_id,
        "product_category": category,
        "product_name": f"{brand} {model_name}",
        "brand": brand,
        "model_number": model_number,
        "serial_number": serial,
        "purchase_date": purchase_dt.isoformat(),
        "purchase_price": price,
        "retailer": random.choice(AUTHORIZED_RETAILERS),
        "warranty_duration_months": warranty_months,
        "warranty_start_date": warranty_start_dt.isoformat(),
        "warranty_expiry_date": warranty_expiry_dt.isoformat(),
        "fault_occurrence_date": fault_dt.isoformat(),
        "fault_description": fault_desc,
        "damage_type": damage_type,
        "claim_submission_date": claim_dt.isoformat(),
        "repair_history": repair_hist,
        "has_receipt": has_receipt,
        "has_warranty_card": has_warranty_card,
        "has_product_image": has_product_image,
        "has_serial_evidence": has_serial_evidence,
        "serial_number_on_receipt": receipt_serial,
        "prior_replacement": prior_replacement,
        "class_label": "Invalid Claim",
    }


def generate_manual_review_claim(claim_idx: int) -> Dict[str, Any]:
    """
    GENERATION LOGIC FOR MANUAL REVIEW CLAIMS (Borderline & Triage Scenarios):
    - Subtype A: 1-2 missing documents (e.g., missing warranty card or blurry serial photo).
    - Subtype B: Boundary date near expiry (fault 1-7 days before expiry, or claim filed in grace period).
    - Subtype C: Plausible unauthorized repair history or high repair frequency (lemon review).
    - Subtype D: Minor contradictions / OCR transcription typo on serial number receipt.
    """
    category, brand, model_name, model_number, product_id, price, warranty_months = generate_product_info()
    serial = generate_serial_number(category, brand)

    review_subtype = random.choice([
        "MISSING_1_OR_2_DOCS",
        "NEAR_EXPIRY_OR_GRACE",
        "UNAUTHORIZED_BUT_PLAUSIBLE_REPAIR",
        "MINOR_SERIAL_OCR_TYPO",
        "SLIGHT_REPORTING_DELAY",
    ])

    purchase_days_ago = random.randint(60, min(700, warranty_months * 30 + 10))
    purchase_dt = date.today() - timedelta(days=purchase_days_ago)
    warranty_start_dt = purchase_dt + timedelta(days=1)
    warranty_expiry_dt = warranty_start_dt + timedelta(days=int(warranty_months * 30.4375))

    # Base values
    has_receipt = True
    has_warranty_card = True
    has_product_image = True
    has_serial_evidence = True
    receipt_serial = serial
    prior_replacement = False
    repair_hist = "0 repairs"
    fault_desc, damage_type = random.choice(CATEGORIES_CONFIG[category]["covered_faults"])

    # Standard fault within window by default
    fault_dt = warranty_start_dt + timedelta(days=random.randint(15, max(20, (warranty_expiry_dt - warranty_start_dt).days - 20)))
    claim_dt = fault_dt + timedelta(days=random.randint(2, 5))

    if review_subtype == "MISSING_1_OR_2_DOCS":
        # Receipt is present, but 1 or 2 supporting attachments are missing
        missing_pattern = random.choice([
            {"card": False, "img": True, "serial_ev": True},
            {"card": True, "img": False, "serial_ev": True},
            {"card": True, "img": True, "serial_ev": False},
            {"card": False, "img": True, "serial_ev": False},
        ])
        has_warranty_card = missing_pattern["card"]
        has_product_image = missing_pattern["img"]
        has_serial_evidence = missing_pattern["serial_ev"]

    elif review_subtype == "NEAR_EXPIRY_OR_GRACE":
        # Fault occurred 1 to 5 days before warranty expired, or claim filed in grace period (1-10 days after)
        case = random.choice(["just_before_expiry", "in_grace_period"])
        if case == "just_before_expiry":
            fault_dt = warranty_expiry_dt - timedelta(days=random.randint(1, 5))
            claim_dt = warranty_expiry_dt + timedelta(days=random.randint(1, 4))
        else:
            fault_dt = warranty_expiry_dt + timedelta(days=random.randint(1, 6))  # In statutory grace period
            claim_dt = fault_dt + timedelta(days=random.randint(1, 4))

    elif review_subtype == "UNAUTHORIZED_BUT_PLAUSIBLE_REPAIR":
        # Unofficial repair history noted (e.g. minor hinge tightening or button fix), or Lemon threshold (3 prior)
        repair_hist = random.choice([
            "1 repair (Unauthorized Third-Party)",
            "2 repairs (1 Authorized, 1 Unauthorized)",
            "3 repairs (Authorized Service Center)",  # Lemon review threshold
        ])
        fault_desc = f"{fault_desc} (Customer notes prior minor servicing at local shop)"

    elif review_subtype == "MINOR_SERIAL_OCR_TYPO":
        # Receipt serial has a single OCR transcription typo (e.g., '0' instead of 'O' or '8' instead of 'B')
        receipt_serial = apply_minor_transcription_typo(serial)

    else:  # SLIGHT_REPORTING_DELAY
        # Filed slightly late (e.g., 9 to 16 days after incident instead of 7 days, due to travel/illness)
        reporting_delay = random.randint(9, 16)
        claim_dt = fault_dt + timedelta(days=reporting_delay)

    return {
        "claim_id": f"CLM-2026-{claim_idx:05d}",
        "user_id": f"USR-{random.randint(10000, 99999)}",
        "product_id": product_id,
        "product_category": category,
        "product_name": f"{brand} {model_name}",
        "brand": brand,
        "model_number": model_number,
        "serial_number": serial,
        "purchase_date": purchase_dt.isoformat(),
        "purchase_price": price,
        "retailer": random.choice(AUTHORIZED_RETAILERS),
        "warranty_duration_months": warranty_months,
        "warranty_start_date": warranty_start_dt.isoformat(),
        "warranty_expiry_date": warranty_expiry_dt.isoformat(),
        "fault_occurrence_date": fault_dt.isoformat(),
        "fault_description": fault_desc,
        "damage_type": damage_type,
        "claim_submission_date": claim_dt.isoformat(),
        "repair_history": repair_hist,
        "has_receipt": has_receipt,
        "has_warranty_card": has_warranty_card,
        "has_product_image": has_product_image,
        "has_serial_evidence": has_serial_evidence,
        "serial_number_on_receipt": receipt_serial,
        "prior_replacement": prior_replacement,
        "class_label": "Manual Review",
    }


# ==============================================================================
# CONTROLLED NOISE INJECTION
# ==============================================================================

def inject_noise_into_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    INJECT CONTROLLED REALISTIC NOISE (~10% rate within each class):
    Applies realistic anomalies that mirror real-world customer service messiness:
    - Valid Claims: Clerical filing typo (submission logged 1 day prior to fault),
      single-character OCR typo on receipt while valid, or missing secondary photo.
    - Invalid Claims: Ambiguous damage categorization (damage_type listed as 'Display Glitch'
      while description text clearly mentions water drop), or missing retailer details.
    - Manual Review: Missing optional fields (NaN / None), price anomalies, or
      inconsistent serial formatting (whitespace or lowercase).
    """
    rec = dict(record)
    label = rec["class_label"]
    noise_type = "None"

    if label == "Valid Claim":
        noise_variant = random.choice([
            "CLERICAL_DATE_TYPO",
            "RECEIPT_OCR_TYPO",
            "MISSING_AUXILIARY_PHOTO",
            "SERIAL_WHITESPACE",
        ])
        if noise_variant == "CLERICAL_DATE_TYPO":
            # Clerical intake error: submission date recorded 1 day prior to fault date due to timezone
            fault_d = date.fromisoformat(rec["fault_occurrence_date"])
            rec["claim_submission_date"] = (fault_d - timedelta(days=1)).isoformat()
            noise_type = "Valid: Clerical Date Misalignment (Submission recorded prior to Fault)"
        elif noise_variant == "RECEIPT_OCR_TYPO":
            # Valid claim where dealer handwritten receipt has 1-char OCR discrepancy
            rec["serial_number_on_receipt"] = apply_minor_transcription_typo(rec["serial_number"])
            noise_type = "Valid: Minor Receipt Serial OCR Discrepancy"
        elif noise_variant == "MISSING_AUXILIARY_PHOTO":
            # Customer brought physical device to center in person; app image checkbox empty
            rec["has_product_image"] = False
            noise_type = "Valid: Missing In-App Photo (Direct In-Store Dropoff)"
        else:
            # Subtle whitespace in serial string from barcode copy-paste
            rec["serial_number"] = f" {rec['serial_number']} "
            noise_type = "Valid: Serial Barcode String Trailing Whitespace"

    elif label == "Invalid Claim":
        noise_variant = random.choice([
            "CONTRADICTORY_DAMAGE_LABEL",
            "RETAILER_METADATA_MISSING",
            "ALL_DOCS_CHECKED_BUT_EXPIRED",
        ])
        if noise_variant == "CONTRADICTORY_DAMAGE_LABEL":
            # Submitter categorized as benign 'Hardware Glitch', but description admits dropped in tea/water
            rec["damage_type"] = "Hardware Glitch / Unspecified Defect"
            rec["fault_description"] = "Device fell into water container; now will not boot or respond to charger"
            noise_type = "Invalid: Contradictory Damage Label vs Narrative Description"
        elif noise_variant == "RETAILER_METADATA_MISSING":
            # Retailer unknown or missing from informal grey market purchase
            rec["retailer"] = None
            noise_type = "Invalid: Missing Retailer Name (Grey Channel)"
        else:
            # Customer ticked all document boxes even though warranty was expired 2 years ago
            rec["has_receipt"] = True
            rec["has_warranty_card"] = True
            rec["has_product_image"] = True
            rec["has_serial_evidence"] = True
            noise_type = "Invalid: All Docs Complete on Out-of-Warranty Unit"

    else:  # Manual Review
        noise_variant = random.choice([
            "MISSING_RECEIPT_SERIAL_VALUE",
            "DISCOUNT_PROMO_PRICE_OUTLIER",
            "SERIAL_LOWERCASE_TYPO",
        ])
        if noise_variant == "MISSING_RECEIPT_SERIAL_VALUE":
            # Receipt serial field left blank by claimant
            rec["serial_number_on_receipt"] = None
            noise_type = "Manual Review: Null Receipt Serial Field"
        elif noise_variant == "DISCOUNT_PROMO_PRICE_OUTLIER":
            # High voucher discount / zero promotional prize price
            rec["purchase_price"] = 0.0
            rec["fault_description"] += " (Promotional raffle prize bundle)"
            noise_type = "Manual Review: Zero Purchase Price Promotional Item"
        else:
            rec["serial_number_on_receipt"] = str(rec["serial_number_on_receipt"]).lower()
            noise_type = "Manual Review: Lowercase Receipt Serial String"

    return rec, noise_type


# ==============================================================================
# MAIN DATASET GENERATION PIPELINE
# ==============================================================================

def generate_full_claims_dataset(
    n_per_class: int = 500,
    noise_ratio: float = 0.10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates balanced synthetic dataset with controlled noise injection.
    """
    total_records = n_per_class * 3
    print(f"[*] Generating {total_records} claims ({n_per_class} per class)...")

    records = []
    noise_stats = {"Valid Claim": 0, "Invalid Claim": 0, "Manual Review": 0, "details": []}

    current_id = 1
    # 1. Generate Valid Claims
    for _ in range(n_per_class):
        rec = generate_valid_claim(current_id)
        if random.random() < noise_ratio:
            rec, n_type = inject_noise_into_record(rec)
            noise_stats["Valid Claim"] += 1
            noise_stats["details"].append({"claim_id": rec["claim_id"], "class": "Valid Claim", "noise": n_type})
        records.append(rec)
        current_id += 1

    # 2. Generate Invalid Claims
    for _ in range(n_per_class):
        rec = generate_invalid_claim(current_id)
        if random.random() < noise_ratio:
            rec, n_type = inject_noise_into_record(rec)
            noise_stats["Invalid Claim"] += 1
            noise_stats["details"].append({"claim_id": rec["claim_id"], "class": "Invalid Claim", "noise": n_type})
        records.append(rec)
        current_id += 1

    # 3. Generate Manual Review Claims
    for _ in range(n_per_class):
        rec = generate_manual_review_claim(current_id)
        if random.random() < noise_ratio:
            rec, n_type = inject_noise_into_record(rec)
            noise_stats["Manual Review"] += 1
            noise_stats["details"].append({"claim_id": rec["claim_id"], "class": "Manual Review", "noise": n_type})
        records.append(rec)
        current_id += 1

    df = pd.DataFrame(records)

    # Shuffle the dataset thoroughly
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"[+] Successfully constructed raw DataFrame with shape: {df.shape}")

    return df, noise_stats


def perform_stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs an exact stratified 70/15/15 split across class_label.
    """
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for label, group in df.groupby("class_label"):
        group_shuffled = group.sample(frac=1.0, random_state=SEED)
        n = len(group_shuffled)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure exact balance
        n_test = n - n_train - n_val

        train_part = group_shuffled.iloc[:n_train]
        val_part = group_shuffled.iloc[n_train:n_train + n_val]
        test_part = group_shuffled.iloc[n_train + n_val:]

        train_dfs.append(train_part)
        val_dfs.append(val_part)
        test_dfs.append(test_part)

    train_df = pd.concat(train_dfs).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_df = pd.concat(val_dfs).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    test_df = pd.concat(test_dfs).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    return train_df, val_df, test_df


def compute_split_summary(split_name: str, split_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes distribution and missing value metrics for a split."""
    class_counts = split_df["class_label"].value_counts().to_dict()
    missing_counts = {col: int(count) for col, count in split_df.isnull().sum().items() if count > 0}

    return {
        "split_name": split_name,
        "record_count": int(len(split_df)),
        "class_balance": class_counts,
        "missing_value_counts": missing_counts,
        "product_categories": split_df["product_category"].value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate AssureX Synthetic Warranty Claims Dataset")
    parser.add_argument("--n-per-class", type=int, default=500, help="Number of records per class (default: 500)")
    parser.add_argument("--noise-ratio", type=float, default=0.10, help="Ratio of controlled noise (default: 0.10)")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output directory for CSVs and JSON")
    args = parser.parse_args()

    # Generate full dataset
    df, noise_stats = generate_full_claims_dataset(
        n_per_class=args.n_per_class,
        noise_ratio=args.noise_ratio,
    )

    # Perform stratified split
    train_df, val_df, test_df = perform_stratified_split(df)

    print(f"\n[+] Split Distribution:")
    print(f"    Train: {len(train_df)} rows")
    print(f"    Val:   {len(val_df)} rows")
    print(f"    Test:  {len(test_df)} rows")

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    stats_path = os.path.join(output_dir, "dataset_stats.json")

    # Save CSV files
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[+] Saved CSVs to {output_dir}/ (train.csv, val.csv, test.csv)")

    # Also save root copies if output_dir is not root
    if output_dir not in (".", ""):
        train_df.to_csv("train.csv", index=False)
        val_df.to_csv("val.csv", index=False)
        test_df.to_csv("test.csv", index=False)
        print("[+] Also saved root copies: train.csv, val.csv, test.csv")

    # Build comprehensive stats summary
    stats_summary = {
        "dataset_metadata": {
            "title": "AssureX Consumer Electronics Warranty Claims Classification Dataset",
            "total_records": int(len(df)),
            "generation_seed": SEED,
            "fields_count": int(df.shape[1]),
            "fields_list": list(df.columns),
            "target_classes": ["Valid Claim", "Invalid Claim", "Manual Review"],
            "overall_class_balance": df["class_label"].value_counts().to_dict(),
        },
        "noise_injection_summary": {
            "target_noise_ratio": args.noise_ratio,
            "total_noisy_records": sum([noise_stats["Valid Claim"], noise_stats["Invalid Claim"], noise_stats["Manual Review"]]),
            "noisy_records_per_class": {
                "Valid Claim": noise_stats["Valid Claim"],
                "Invalid Claim": noise_stats["Invalid Claim"],
                "Manual Review": noise_stats["Manual Review"],
            },
            "noise_types_sample": noise_stats["details"][:10],
        },
        "splits": {
            "train": compute_split_summary("train", train_df),
            "val": compute_split_summary("val", val_df),
            "test": compute_split_summary("test", test_df),
        },
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)
    print(f"[+] Saved summary stats to {stats_path}")

    # Also save root copy of dataset_stats.json if output_dir is not root
    if output_dir not in (".", ""):
        with open("dataset_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats_summary, f, indent=2)
        print("[+] Also saved root copy: dataset_stats.json")

    print("\n[SUCCESS] Dataset generation and stratified splitting complete!")


if __name__ == "__main__":
    main()
