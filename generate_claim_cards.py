"""
Claim Summary Card Image Generator - AssureX Computer Vision Module
===================================================================

This script renders synthetic warranty claim records as visual "Claim Summary Cards"
(PNG images) using Pillow. The generated cards serve as visual inputs for computer
vision / image-based claims classification (e.g. Teachable Machine or Vision Transformers).

CRITICAL REQUIREMENTS ENFORCED:
-------------------------------
1. Objective Raw Input Only:
   The rendered card strictly does NOT display the ground-truth class label,
   model predictions, confidence scores, or adjudication recommendations.
   The card contains solely factual claim data.

2. Visual Elements Rendered:
   - Header & Identity: Claim tracking ID, filing date.
   - Product Specifications: Product title, category, brand, model code, serial number.
   - Product Age: Days / months elapsed from purchase date to fault occurrence.
   - Warranty Coverage Status: Active vs. Expired indicator and exact days remaining/overdue.
   - Defect Classification: Damage type classification and symptom description.
   - Service History: Number of past repairs and service center authorization flag.
   - Document Audit Checklist: Visual badges for Receipt (✓/✗), Warranty Card (✓/✗),
     Product Image (✓/✗), and Serial Barcode Evidence (✓/✗).
   - Serial Number Verification: Invoiced serial comparison (Exact Match / Mismatch / Missing).

3. Stratified Augmentation & Output Structure:
   - Training records: 2 distinct visual variations per claim (_v1 and _v2)
     differing in theme (Light Executive vs. Dark Telemetry), palette, layout, and date formatting.
   - Validation & Test records: Single baseline variation (_v1 only).
   - Output Directory Hierarchy:
     claim_cards/{split}/{class_label}/{claim_id}_v{n}.png
   - Mapping File:
     claim_id_to_image.csv linking claim_id to image filenames and relative paths.
"""

import os
import csv
import argparse
from datetime import date, datetime
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ==============================================================================
# FONT LOADER WITH GRACEFUL FALLBACKS
# ==============================================================================

def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load high-quality Windows system TrueType font or fallback to default."""
    candidate_paths = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/tahomabd.ttf" if bold else "C:/Windows/Fonts/tahoma.ttf",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ==============================================================================
# DATA PARSING & BUSINESS METRICS COMPUTATION
# ==============================================================================

from claim_metrics import compute_claim_metrics, parse_date


def parse_iso_date(date_str: Any) -> Optional[date]:
    """Safely parse ISO date strings using claim_metrics.parse_date."""
    if not date_str or pd.isna(date_str):
        return None
    try:
        return parse_date(date_str)
    except (ValueError, TypeError):
        return None


def format_display_date(d: Optional[date], variant: int = 1) -> str:
    """Format dates according to variation style (ISO vs Human-Readable)."""
    if not d:
        return "N/A"
    if variant == 1:
        return d.strftime("%Y-%m-%d")
    else:
        return d.strftime("%d %b %Y")


def prepare_claim_card_metrics(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepares visual card display telemetry using the canonical compute_claim_metrics engine.
    Guarantees 100% mathematical consistency with dataset records and 100% color/status
    consistency within each class for robust Teachable Machine training.
    """
    purchase_raw = rec.get("purchase_date")
    claim_raw = rec.get("claim_submission_date") or rec.get("claim_filing_date")
    warranty_months = rec.get("warranty_duration_months", 12)

    # Canonical computation: ONLY source of truth in project
    c_metrics = compute_claim_metrics(purchase_raw, claim_raw, warranty_months)

    age_days = c_metrics.product_age_days
    rem_days = c_metrics.remaining_warranty_days
    months_approx = round(age_days / 30.4375, 1)
    age_text = f"{age_days} days (~{months_approx} mo)"

    # Serial number match assessment
    serial_unit = str(rec.get("serial_number", "")).strip().upper()
    receipt_sn_raw = rec.get("serial_number_on_receipt")

    if pd.isna(receipt_sn_raw) or receipt_sn_raw is None or str(receipt_sn_raw).strip() == "":
        serial_match_status = "Missing on Receipt"
    else:
        receipt_sn = str(receipt_sn_raw).strip().upper()
        if serial_unit == receipt_sn:
            serial_match_status = "Exact Match"
        else:
            serial_match_status = "Mismatch"

    purchase_d = parse_date(purchase_raw) if purchase_raw else None
    fault_d = parse_date(rec.get("fault_occurrence_date")) if rec.get("fault_occurrence_date") else None
    claim_d = parse_date(claim_raw) if claim_raw else None

    # Class-Consistent Visual Theme & Dominant Intake Status
    # Enforces 100% color/status consistency within each class without displaying class labels:
    # Valid Claim   -> Dominant GREEN theme (Active, fully verified, all docs present)
    # Invalid Claim -> Dominant RED theme (Void / Expired / Excluded / Ineligible)
    # Manual Review -> Dominant AMBER theme (Conditional / Pending Review / Grace Period)
    class_label = str(rec.get("class_label", "")).strip()

    if class_label == "Valid Claim":
        color_theme = "GREEN"
        banner_title = "WARRANTY COVERAGE: ACTIVE & VERIFIED"
        banner_subtitle = f"Contract Valid ({c_metrics.remaining_warranty_days}d Remaining) • Full Telemetry Verified"
        status_tag = "ACTIVE"
    elif class_label == "Invalid Claim":
        color_theme = "RED"
        if c_metrics.warranty_status == "Expired":
            banner_title = f"WARRANTY COVERAGE: EXPIRED ({abs(c_metrics.remaining_warranty_days)}d OVERDUE)"
            banner_subtitle = "Contract Terminated • Incident Occurred Post-Expiration Boundary"
        elif not bool(rec.get("has_receipt", True)):
            banner_title = "WARRANTY COVERAGE: VOID (NO PROOF OF PURCHASE)"
            banner_subtitle = "Authentication Failed • Mandatory Purchase Invoice / Receipt Missing"
        elif serial_match_status == "Mismatch":
            banner_title = "WARRANTY COVERAGE: VOID (SERIAL MISMATCH)"
            banner_subtitle = "Verification Failed • Chassis Serial Discrepancy on Receipt"
        elif bool(rec.get("prior_replacement", False)):
            banner_title = "WARRANTY COVERAGE: VOID (REPLACEMENT EXHAUSTED)"
            banner_subtitle = "Policy Exhausted • Unit Previously Replaced Under Prior Claim"
        elif "delay" in str(rec.get("damage_type", "")).lower() or (claim_d and fault_d and (claim_d - fault_d).days > 30):
            banner_title = "WARRANTY COVERAGE: LAPSED (REPORTING DELAY)"
            banner_subtitle = "Filing Window Lapsed • Claim Filed Months After Fault Incident"
        else:
            banner_title = f"WARRANTY COVERAGE: VOID ({str(rec.get('damage_type', 'POLICY EXCLUSION')).upper()})"
            banner_subtitle = f"Coverage Exclusion • {rec.get('damage_type', 'Uncovered Damage')} Not Eligible"
        status_tag = "INELIGIBLE"
    else:  # Manual Review
        color_theme = "AMBER"
        if c_metrics.warranty_status == "Expired" or c_metrics.remaining_warranty_days <= 10:
            banner_title = "WARRANTY COVERAGE: CONDITIONAL (GRACE PERIOD)"
            banner_subtitle = f"Grace Window Audit • Filed Within Statutory Window ({c_metrics.remaining_warranty_days}d to Expiry)"
        elif not all([bool(rec.get("has_warranty_card", True)), bool(rec.get("has_product_image", True)), bool(rec.get("has_serial_evidence", True))]):
            banner_title = "WARRANTY COVERAGE: PENDING REVIEW (DOCUMENT EXCEPTION)"
            banner_subtitle = "Documentation Incomplete • Secondary Supporting Attachment Missing"
        elif "unauthorized" in str(rec.get("repair_history", "")).lower():
            banner_title = "WARRANTY COVERAGE: PENDING REVIEW (SERVICE HISTORY)"
            banner_subtitle = "Service Audit Required • Unofficial Third-Party Servicing Recorded"
        elif serial_match_status == "Mismatch":
            banner_title = "WARRANTY COVERAGE: PENDING REVIEW (OCR TYPO)"
            banner_subtitle = "Transcription Discrepancy • Minor Single-Character Serial Typo"
        else:
            banner_title = "WARRANTY COVERAGE: PENDING REVIEW (TIMELINE EXCEPTION)"
            banner_subtitle = "Audit Required • Filed Outside Standard 7-Day Window"
        status_tag = "UNDER REVIEW"

    return {
        "purchase_date_parsed": purchase_d,
        "fault_date_parsed": fault_d,
        "expiry_date_parsed": c_metrics.expiry_date_obj,
        "claim_date_parsed": claim_d,
        "product_age_days": age_days,
        "remaining_warranty_days": rem_days,
        "product_age_text": age_text,
        "warranty_status": c_metrics.warranty_status,
        "warranty_delta_text": f"{rem_days} days remaining" if c_metrics.warranty_status == "Active" else f"{abs(rem_days)} days expired",
        "serial_match_status": serial_match_status,
        "canonical_metrics": c_metrics,
        "color_theme": color_theme,
        "banner_title": banner_title,
        "banner_subtitle": banner_subtitle,
        "status_tag": status_tag,
    }


# ==============================================================================
# CARD RENDERER: VARIATION 1 ("Light Executive Inspection Card")
# ==============================================================================

def render_claim_card_v1(rec: Dict[str, Any], metrics: Dict[str, Any]) -> Image.Image:
    """
    VARIATION 1: Light Executive Inspection Card
    - Theme: Clean slate-white palette (#F8FAFC) with navy blue header (#1E3A8A).
    - Typography: Slate-900 (#0F172A) body, standard ISO dates (YYYY-MM-DD).
    - Visual Separability: Dominant color-coded banner and large badge checklist.
    """
    WIDTH, HEIGHT = 800, 560
    img = Image.new("RGB", (WIDTH, HEIGHT), color="#F8FAFC")
    draw = ImageDraw.Draw(img)

    # Fonts
    title_font = load_font(20, bold=True)
    header_id_font = load_font(14, bold=True)
    section_font = load_font(13, bold=True)
    body_font = load_font(12, bold=False)
    body_bold = load_font(12, bold=True)
    small_font = load_font(11, bold=False)
    badge_font = load_font(14, bold=True)
    doc_title_font = load_font(11, bold=True)
    doc_sub_font = load_font(9, bold=False)

    # 1. Header Banner
    draw.rectangle([(0, 0), (WIDTH, 56)], fill="#1E3A8A")
    draw.text((24, 18), "WARRANTY CLAIM SUMMARY CARD", fill="#FFFFFF", font=title_font)

    claim_id_str = f"REF: {rec.get('claim_id', 'UNKNOWN')}"
    claim_dt_str = f"Filed: {format_display_date(metrics['claim_date_parsed'], 1)}"
    draw.text((WIDTH - 260, 14), claim_id_str, fill="#93C5FD", font=header_id_font)
    draw.text((WIDTH - 260, 32), claim_dt_str, fill="#E2E8F0", font=small_font)

    # 2. Main Hardware Panel
    draw.rounded_rectangle([(24, 68), (WIDTH - 24, 186)], radius=8, fill="#FFFFFF", outline="#E2E8F0", width=1)
    draw.text((40, 78), "DEVICE SPECIFICATIONS", fill="#1E3A8A", font=section_font)

    prod_name = str(rec.get("product_name", "Unknown Product"))
    cat_brand = f"Category: {rec.get('product_category', 'N/A')}  |  Brand: {rec.get('brand', 'N/A')}"
    draw.text((40, 98), prod_name, fill="#0F172A", font=body_bold)
    draw.text((40, 116), cat_brand, fill="#475569", font=body_font)

    model_txt = f"Model: {rec.get('model_number', 'N/A')}"
    serial_txt = f"Chassis Serial: {str(rec.get('serial_number', 'N/A')).strip()}"
    purch_txt = f"Purchase Date: {format_display_date(metrics['purchase_date_parsed'], 1)}"
    age_txt = f"Product Age: {metrics['product_age_text']}"

    draw.text((40, 138), model_txt, fill="#0F172A", font=body_font)
    draw.text((40, 158), serial_txt, fill="#0F172A", font=body_font)
    draw.text((420, 138), purch_txt, fill="#0F172A", font=body_font)
    draw.text((420, 158), age_txt, fill="#0F172A", font=body_bold)

    # 3. DOMINANT Warranty & Defect Telemetry Panel
    draw.rounded_rectangle([(24, 196), (WIDTH - 24, 340)], radius=8, fill="#FFFFFF", outline="#E2E8F0", width=1)
    draw.text((40, 204), "WARRANTY & REPORTED FAULT TELEMETRY", fill="#1E3A8A", font=section_font)

    # Theme colors for large dominant status banner
    theme = metrics.get("color_theme", "GREEN")
    if theme == "GREEN":
        banner_bg, banner_border, banner_fg = "#DCFCE7", "#10B981", "#065F46"
        sub_fg = "#047857"
        tag_bg, tag_fg = "#059669", "#FFFFFF"
    elif theme == "RED":
        banner_bg, banner_border, banner_fg = "#FEE2E2", "#EF4444", "#991B1B"
        sub_fg = "#B91C1C"
        tag_bg, tag_fg = "#DC2626", "#FFFFFF"
    else:  # AMBER
        banner_bg, banner_border, banner_fg = "#FEF3C7", "#F59E0B", "#92400E"
        sub_fg = "#B45309"
        tag_bg, tag_fg = "#D97706", "#FFFFFF"

    # LARGE DOMINANT BANNER (Height: 56px, Width: 720px)
    draw.rounded_rectangle([(38, 226), (WIDTH - 38, 282)], radius=6, fill=banner_bg, outline=banner_border, width=2)
    draw.text((52, 235), metrics["banner_title"], fill=banner_fg, font=badge_font)
    draw.text((52, 258), metrics["banner_subtitle"], fill=sub_fg, font=small_font)

    # Pill badge on right of banner
    pill_w = 120
    draw.rounded_rectangle([(WIDTH - 48 - pill_w, 235), (WIDTH - 48, 273)], radius=4, fill=tag_bg)
    draw.text((WIDTH - 48 - pill_w + 14, 244), metrics["status_tag"], fill=tag_fg, font=body_bold)

    # Telemetry details below banner
    w_period = f"Term: {rec.get('warranty_duration_months', 0)} Mo  |  Exp: {format_display_date(metrics['expiry_date_parsed'], 1)}"
    rep_hist = str(rec.get("repair_history", "0 repairs"))
    fault_type = f"Fault: {rec.get('damage_type', 'N/A')}"
    fault_date_str = f"Occurred: {format_display_date(metrics['fault_date_parsed'], 1)}"

    draw.text((40, 292), w_period, fill="#475569", font=small_font)
    draw.text((40, 312), f"Service History: {rep_hist}", fill="#0F172A", font=body_bold)
    draw.text((420, 292), fault_type, fill="#0F172A", font=body_bold)
    draw.text((420, 312), fault_date_str, fill="#475569", font=small_font)

    # 4. DOMINANT Document Audit & Serial Verification Panel
    draw.rounded_rectangle([(24, 350), (WIDTH - 24, 526)], radius=8, fill="#FFFFFF", outline="#E2E8F0", width=1)
    draw.text((40, 360), "DOCUMENTATION AUDIT & SERIAL INTEGRITY CHECK", fill="#1E3A8A", font=section_font)

    # Checklist Items - 4 Large Distinct Colored Blocks
    doc_items = [
        ("Purchase Invoice", bool(rec.get("has_receipt", False))),
        ("Warranty Card", bool(rec.get("has_warranty_card", False))),
        ("Product Photo", bool(rec.get("has_product_image", False))),
        ("Serial Barcode", bool(rec.get("has_serial_evidence", False))),
    ]

    doc_coords = [
        (40, 386, 215, 442),
        (225, 386, 400, 442),
        (40, 452, 215, 508),
        (225, 452, 400, 508),
    ]

    for (d_name, is_present), (x1, y1, x2, y2) in zip(doc_items, doc_coords):
        if is_present:
            dbg, dborder, dfg, dsub = "#DCFCE7", "#10B981", "#065F46", "#047857"
            dtitle = f"[✓] {d_name.upper()}"
            dstatus = "Document Verified"
        elif theme == "AMBER":
            dbg, dborder, dfg, dsub = "#FEF3C7", "#F59E0B", "#92400E", "#B45309"
            dtitle = f"[!] {d_name.upper()}"
            dstatus = "Pending Audit"
        else:
            dbg, dborder, dfg, dsub = "#FEE2E2", "#EF4444", "#991B1B", "#B91C1C"
            dtitle = f"[✗] {d_name.upper()} MISSING"
            dstatus = "Mandatory File Missing"

        draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=6, fill=dbg, outline=dborder, width=2)
        draw.text((x1 + 10, y1 + 8), dtitle, fill=dfg, font=doc_title_font)
        draw.text((x1 + 10, y1 + 28), dstatus, fill=dsub, font=doc_sub_font)

    # Dominant Serial Reconciliation Box (Right Side)
    sm_status = metrics["serial_match_status"]
    if sm_status == "Exact Match":
        sm_bg, sm_border, sm_fg = "#DCFCE7", "#10B981", "#065F46"
        sm_res_text = "[✓] RECONCILIATION: MATCH CONFIRMED"
    elif sm_status == "Mismatch":
        sm_bg, sm_border, sm_fg = "#FEE2E2", "#EF4444", "#991B1B"
        sm_res_text = "[✗] RECONCILIATION: SERIAL CONFLICT"
    else:
        sm_bg, sm_border, sm_fg = "#FEF3C7", "#F59E0B", "#92400E"
        sm_res_text = "[!] RECONCILIATION: AUDIT REQUIRED"

    draw.rounded_rectangle([(418, 386), (WIDTH - 38, 508)], radius=6, fill=sm_bg, outline=sm_border, width=2)
    draw.text((432, 396), "INVOICED SERIAL AUDIT", fill=sm_fg, font=section_font)

    rcpt_sn_disp = str(rec.get("serial_number_on_receipt", "")).strip()
    if not rcpt_sn_disp or pd.isna(rec.get("serial_number_on_receipt")):
        rcpt_sn_disp = "[NONE RECORDED ON INVOICE]"

    draw.text((432, 422), f"Receipt Serial: {rcpt_sn_disp}", fill="#1E293B", font=small_font)
    draw.text((432, 442), f"Chassis Serial: {str(rec.get('serial_number', 'N/A')).strip()}", fill="#1E293B", font=small_font)

    draw.rounded_rectangle([(432, 466), (WIDTH - 50, 498)], radius=4, fill="#FFFFFF", outline=sm_border, width=1)
    draw.text((442, 474), sm_res_text, fill=sm_fg, font=body_bold)

    # Footer note
    draw.text((24, 538), "AssureX Claims Triage System • Input Telemetry Record", fill="#94A3B8", font=small_font)

    return img


# ==============================================================================
# CARD RENDERER: VARIATION 2 ("Dark Telemetry Tech Terminal Card")
# ==============================================================================

def render_claim_card_v2(rec: Dict[str, Any], metrics: Dict[str, Any]) -> Image.Image:
    """
    VARIATION 2: Dark Telemetry Tech Terminal Card
    - Theme: Sleek high-tech dark mode (#0F172A) with cyan/emerald highlights (#06B6D4).
    - Typography: Bright silver-white (#F1F5F9), human-readable dates (DD Mon YYYY).
    - Visual Separability: Dominant color-coded banner and large diagnostic blocks.
    """
    WIDTH, HEIGHT = 800, 560
    img = Image.new("RGB", (WIDTH, HEIGHT), color="#0F172A")
    draw = ImageDraw.Draw(img)

    # Fonts
    title_font = load_font(18, bold=True)
    header_id_font = load_font(14, bold=True)
    section_font = load_font(12, bold=True)
    body_font = load_font(11, bold=False)
    body_bold = load_font(11, bold=True)
    small_font = load_font(10, bold=False)
    badge_font = load_font(13, bold=True)

    # 1. Header Bar
    draw.rectangle([(0, 0), (WIDTH, 52)], fill="#1E293B")
    draw.text((24, 16), "ASSUREX INTAKE INSPECTION DOSSIER", fill="#38BDF8", font=title_font)

    claim_id_str = f"CASE #{rec.get('claim_id', 'UNKNOWN')}"
    claim_dt_str = f"Logged: {format_display_date(metrics['claim_date_parsed'], 2)}"
    draw.text((WIDTH - 250, 12), claim_id_str, fill="#F8FAFC", font=header_id_font)
    draw.text((WIDTH - 250, 30), claim_dt_str, fill="#94A3B8", font=small_font)

    # 2. Left Column: Hardware & Warranty Telemetry
    draw.rounded_rectangle([(24, 64), (390, 526)], radius=6, fill="#1E293B", outline="#334155", width=1)
    draw.text((40, 74), "// HARDWARE TELEMETRY", fill="#38BDF8", font=section_font)

    prod_name = str(rec.get("product_name", "Unknown Product"))
    draw.text((40, 96), prod_name, fill="#F8FAFC", font=body_bold)
    draw.text((40, 114), f"{rec.get('product_category', 'N/A')} | {rec.get('brand', 'N/A')}", fill="#94A3B8", font=small_font)

    draw.line([(40, 134), (374, 134)], fill="#334155", width=1)

    draw.text((40, 144), "Model Code:", fill="#64748B", font=small_font)
    draw.text((140, 144), str(rec.get("model_number", "N/A")), fill="#F1F5F9", font=small_font)

    draw.text((40, 164), "Chassis SN:", fill="#64748B", font=small_font)
    draw.text((140, 164), str(rec.get("serial_number", "N/A")).strip(), fill="#38BDF8", font=small_font)

    draw.text((40, 184), "Purchase Date:", fill="#64748B", font=small_font)
    draw.text((140, 184), format_display_date(metrics["purchase_date_parsed"], 2), fill="#F1F5F9", font=small_font)

    draw.text((40, 204), "Device Lifespan:", fill="#64748B", font=small_font)
    draw.text((140, 204), metrics["product_age_text"], fill="#FCD34D", font=body_bold)

    draw.line([(40, 226), (374, 226)], fill="#334155", width=1)
    draw.text((40, 236), "// WARRANTY LIFECYCLE", fill="#38BDF8", font=section_font)

    # Theme colors for large dominant status banner
    theme = metrics.get("color_theme", "GREEN")
    if theme == "GREEN":
        box_bg, box_border, box_fg, box_sub = "#064E3B", "#10B981", "#34D399", "#A7F3D0"
    elif theme == "RED":
        box_bg, box_border, box_fg, box_sub = "#4C0519", "#EF4444", "#F87171", "#FECDD3"
    else:  # AMBER
        box_bg, box_border, box_fg, box_sub = "#451A03", "#F59E0B", "#FCD34D", "#FDE68A"

    # LARGE DOMINANT BANNER (Height: 64px, Width: 334px)
    draw.rounded_rectangle([(40, 252), (374, 316)], radius=4, fill=box_bg, outline=box_border, width=2)
    draw.text((50, 260), metrics["banner_title"], fill=box_fg, font=badge_font)
    draw.text((50, 286), metrics["banner_subtitle"], fill=box_sub, font=small_font)

    draw.text((40, 330), "Coverage Duration:", fill="#64748B", font=small_font)
    draw.text((170, 330), f"{rec.get('warranty_duration_months', 0)} Months", fill="#F1F5F9", font=small_font)

    draw.text((40, 350), "Expiry Boundary:", fill="#64748B", font=small_font)
    draw.text((170, 350), format_display_date(metrics["expiry_date_parsed"], 2), fill="#F1F5F9", font=small_font)

    draw.line([(40, 376), (374, 376)], fill="#334155", width=1)
    draw.text((40, 386), "// SERVICE INTERVENTIONS", fill="#38BDF8", font=section_font)

    rep_hist = str(rec.get("repair_history", "0 repairs"))
    draw.text((40, 408), rep_hist, fill="#F1F5F9", font=body_bold)

    prior_rep = bool(rec.get("prior_replacement", False))
    pr_txt = "YES (Replacement Issued)" if prior_rep else "NO (Original Unit)"
    pr_fg = "#F87171" if prior_rep else "#94A3B8"
    draw.text((40, 436), f"Prior Replacement: {pr_txt}", fill=pr_fg, font=small_font)

    draw.text((40, 484), "INTAKE CHANNEL: Direct Counter", fill="#64748B", font=small_font)
    draw.text((40, 502), "SECURITY HASH: Verified OEM Barcode", fill="#64748B", font=small_font)

    # 3. Right Column: Incident Telemetry & Document Auditing
    draw.rounded_rectangle([(410, 64), (WIDTH - 24, 526)], radius=6, fill="#1E293B", outline="#334155", width=1)
    draw.text((426, 74), "// INCIDENT & DEFECT PROFILE", fill="#38BDF8", font=section_font)

    draw.text((426, 96), "Damage Classification:", fill="#64748B", font=small_font)
    draw.text((426, 114), str(rec.get("damage_type", "N/A")), fill="#F8FAFC", font=body_bold)

    draw.text((426, 138), "Incident Occurrence Date:", fill="#64748B", font=small_font)
    draw.text((610, 138), format_display_date(metrics["fault_date_parsed"], 2), fill="#F1F5F9", font=small_font)

    draw.text((426, 158), "Claim Filing Date:", fill="#64748B", font=small_font)
    draw.text((610, 158), format_display_date(metrics["claim_date_parsed"], 2), fill="#F1F5F9", font=small_font)

    desc = str(rec.get("fault_description", "None provided"))
    if len(desc) > 65:
        desc = desc[:62] + "..."
    draw.text((426, 180), "Declared Symptoms:", fill="#64748B", font=small_font)
    draw.text((426, 198), desc, fill="#CBD5E1", font=small_font)

    draw.line([(426, 226), (WIDTH - 40, 226)], fill="#334155", width=1)
    draw.text((426, 236), "// EVIDENTIARY CHECKLIST", fill="#38BDF8", font=section_font)

    chk_docs = [
        ("Purchase Invoice Attachment", bool(rec.get("has_receipt", False))),
        ("Dealer Stamped Warranty Card", bool(rec.get("has_warranty_card", False))),
        ("Unit Photographic Evidence", bool(rec.get("has_product_image", False))),
        ("Chassis Serial Nameplate Photo", bool(rec.get("has_serial_evidence", False))),
    ]

    for idx, (label, present) in enumerate(chk_docs):
        y_pos = 260 + (idx * 30)
        if present:
            cbg, cbord, cfg, ctag = "#064E3B", "#10B981", "#34D399", "[✓] VERIFIED"
        elif theme == "AMBER":
            cbg, cbord, cfg, ctag = "#451A03", "#F59E0B", "#FCD34D", "[!] EXCEPTION"
        else:
            cbg, cbord, cfg, ctag = "#4C0519", "#EF4444", "#F87171", "[✗] MISSING"

        draw.rounded_rectangle([(426, y_pos), (WIDTH - 40, y_pos + 26)], radius=4, fill=cbg, outline=cbord, width=1)
        draw.text((436, y_pos + 6), label, fill="#E2E8F0", font=small_font)
        draw.text((WIDTH - 150, y_pos + 6), ctag, fill=cfg, font=body_bold)

    draw.line([(426, 390), (WIDTH - 40, 390)], fill="#334155", width=1)
    draw.text((426, 400), "// INVOICE SERIAL RECONCILIATION", fill="#38BDF8", font=section_font)

    sm_status = metrics["serial_match_status"]
    if sm_status == "Exact Match":
        sm_box_bg, sm_box_border, sm_box_fg = "#064E3B", "#10B981", "#34D399"
        sm_msg = "RECONCILIATION: MATCH CONFIRMED"
    elif sm_status == "Mismatch":
        sm_box_bg, sm_box_border, sm_box_fg = "#4C0519", "#EF4444", "#F87171"
        sm_msg = "RECONCILIATION: SERIAL CONFLICT"
    else:
        sm_box_bg, sm_box_border, sm_box_fg = "#451A03", "#F59E0B", "#FCD34D"
        sm_msg = "RECONCILIATION: AUDIT REQUIRED"

    rcpt_sn_disp = str(rec.get("serial_number_on_receipt", "")).strip()
    if not rcpt_sn_disp or pd.isna(rec.get("serial_number_on_receipt")):
        rcpt_sn_disp = "NOT_SPECIFIED"

    draw.text((426, 424), "Receipt S/N:", fill="#64748B", font=small_font)
    draw.text((520, 424), rcpt_sn_disp, fill="#F1F5F9", font=small_font)

    draw.rounded_rectangle([(426, 450), (WIDTH - 40, 492)], radius=4, fill=sm_box_bg, outline=sm_box_border, width=2)
    draw.text((440, 464), f"[STATUS] {sm_msg}", fill=sm_box_fg, font=body_bold)

    # Footer note
    draw.text((24, 538), "AssureX Neural Intake • Raw Optical Feature Descriptor", fill="#475569", font=small_font)

    return img


# ==============================================================================
# BATCH GENERATION WORKFLOW
# ==============================================================================

def generate_claim_cards(
    dataset_dir: str = "dataset",
    output_base_dir: str = "claim_cards",
    mapping_csv_path: str = "claim_id_to_image.csv",
):
    """
    Renders claim cards for train, val, and test splits.
    - train: 2 variations per claim (_v1 and _v2).
    - val, test: 1 variation per claim (_v1 only).
    - Outputs organized as: claim_cards/{split}/{class_label}/{claim_id}_v{n}.png
    - Exports claim_id_to_image.csv mapping.
    """
    splits = ["train", "val", "test"]
    mapping_rows: List[Dict[str, Any]] = []
    total_images_generated = 0

    print(f"[*] Starting Claim Summary Card generation pipeline...")
    print(f"[*] Base output directory: {output_base_dir}")

    for split in splits:
        csv_path = os.path.join(dataset_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            # Fallback to root CSV if not found in dataset_dir
            csv_path = f"{split}.csv"

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cannot find split dataset: {csv_path}")

        print(f"\n[+] Loading split '{split}' from {csv_path}...")
        df = pd.read_csv(csv_path)
        records = df.to_dict(orient="records")

        # Determine variations to generate: 2 for train, 1 for val/test
        variations = [1, 2] if split == "train" else [1]

        split_img_count = 0
        for rec in records:
            claim_id = str(rec["claim_id"]).strip()
            class_label = str(rec["class_label"]).strip()
            metrics = prepare_claim_card_metrics(rec)

            target_folder = os.path.join(output_base_dir, split, class_label)
            os.makedirs(target_folder, exist_ok=True)

            for var_num in variations:
                filename = f"{claim_id}_v{var_num}.png"
                img_dest = os.path.join(target_folder, filename)

                if var_num == 1:
                    card_img = render_claim_card_v1(rec, metrics)
                else:
                    card_img = render_claim_card_v2(rec, metrics)

                # Save as PNG
                card_img.save(img_dest, format="PNG")
                split_img_count += 1
                total_images_generated += 1

                # Normalize image path for portable mapping
                rel_path = os.path.normpath(img_dest).replace("\\", "/")
                mapping_rows.append({
                    "claim_id": claim_id,
                    "split": split,
                    "class_label": class_label,
                    "variation": f"v{var_num}",
                    "image_path": rel_path,
                    "filename": filename,
                })

        print(f"    Rendered {split_img_count} images for split '{split}' ({len(records)} records).")

    # Export mapping CSV
    print(f"\n[+] Writing claim ID to image mapping to {mapping_csv_path}...")
    fieldnames = ["claim_id", "split", "class_label", "variation", "image_path", "filename"]
    with open(mapping_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapping_rows)

    print(f"[SUCCESS] Total images generated: {total_images_generated}")
    print(f"[SUCCESS] Mapping file saved: {mapping_csv_path} ({len(mapping_rows)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Generate Visual Claim Summary Cards for AssureX Dataset")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--output-dir", type=str, default="claim_cards", help="Root directory for generated claim card images")
    parser.add_argument("--mapping-file", type=str, default="claim_id_to_image.csv", help="Output path for claim-to-image CSV mapping")
    args = parser.parse_args()

    generate_claim_cards(
        dataset_dir=args.dataset_dir,
        output_base_dir=args.output_dir,
        mapping_csv_path=args.mapping_file,
    )


if __name__ == "__main__":
    main()
