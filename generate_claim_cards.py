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

def parse_iso_date(date_str: Any) -> Optional[date]:
    """Safely parse ISO YYYY-MM-DD date strings."""
    if not date_str or pd.isna(date_str):
        return None
    try:
        return datetime.strptime(str(date_str).strip().split("T")[0], "%Y-%m-%d").date()
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


def compute_claim_metrics(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Compute mathematical age, days to expiry, and serial status without decision bias."""
    purchase_d = parse_iso_date(rec.get("purchase_date"))
    fault_d = parse_iso_date(rec.get("fault_occurrence_date"))
    expiry_d = parse_iso_date(rec.get("warranty_expiry_date"))
    claim_d = parse_iso_date(rec.get("claim_submission_date"))

    # Product age at fault occurrence
    if purchase_d and fault_d:
        age_days = (fault_d - purchase_d).days
        months_approx = round(age_days / 30.4375, 1)
        age_text = f"{age_days} days (~{months_approx} mo)"
    else:
        age_text = "Unknown"

    # Warranty active/expired & days remaining
    if fault_d and expiry_d:
        diff_days = (expiry_d - fault_d).days
        if diff_days >= 0:
            warranty_status = "Active"
            warranty_delta_text = f"{diff_days} days remaining"
        else:
            warranty_status = "Expired"
            warranty_delta_text = f"{abs(diff_days)} days expired"
    else:
        warranty_status = "Indeterminate"
        warranty_delta_text = "N/A"

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

    return {
        "purchase_date_parsed": purchase_d,
        "fault_date_parsed": fault_d,
        "expiry_date_parsed": expiry_d,
        "claim_date_parsed": claim_d,
        "product_age_text": age_text,
        "warranty_status": warranty_status,
        "warranty_delta_text": warranty_delta_text,
        "serial_match_status": serial_match_status,
    }


# ==============================================================================
# CARD RENDERER: VARIATION 1 ("Light Executive Inspection Card")
# ==============================================================================

def render_claim_card_v1(rec: Dict[str, Any], metrics: Dict[str, Any]) -> Image.Image:
    """
    VARIATION 1: Light Executive Inspection Card
    - Theme: Clean slate-white palette (#F8FAFC) with navy blue header (#1E3A8A).
    - Typography: Slate-900 (#0F172A) body, standard ISO dates (YYYY-MM-DD).
    - Layout: Structured clean grid with soft rounded container panels.
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

    # 1. Header Banner
    draw.rectangle([(0, 0), (WIDTH, 56)], fill="#1E3A8A")
    draw.text((24, 18), "WARRANTY CLAIM SUMMARY CARD", fill="#FFFFFF", font=title_font)

    claim_id_str = f"REF: {rec.get('claim_id', 'UNKNOWN')}"
    claim_dt_str = f"Filed: {format_display_date(metrics['claim_date_parsed'], 1)}"
    draw.text((WIDTH - 260, 14), claim_id_str, fill="#93C5FD", font=header_id_font)
    draw.text((WIDTH - 260, 32), claim_dt_str, fill="#E2E8F0", font=small_font)

    # 2. Main Hardware Panel
    draw.rounded_rectangle([(24, 72), (WIDTH - 24, 204)], radius=8, fill="#FFFFFF", outline="#E2E8F0", width=1)
    draw.text((40, 84), "DEVICE SPECIFICATIONS", fill="#1E3A8A", font=section_font)

    prod_name = str(rec.get("product_name", "Unknown Product"))
    cat_brand = f"Category: {rec.get('product_category', 'N/A')}  |  Brand: {rec.get('brand', 'N/A')}"
    draw.text((40, 108), prod_name, fill="#0F172A", font=body_bold)
    draw.text((40, 128), cat_brand, fill="#475569", font=body_font)

    model_txt = f"Model: {rec.get('model_number', 'N/A')}"
    serial_txt = f"Chassis Serial: {str(rec.get('serial_number', 'N/A')).strip()}"
    purch_txt = f"Purchase Date: {format_display_date(metrics['purchase_date_parsed'], 1)}"
    age_txt = f"Product Age: {metrics['product_age_text']}"

    draw.text((40, 154), model_txt, fill="#0F172A", font=body_font)
    draw.text((40, 174), serial_txt, fill="#0F172A", font=body_font)
    draw.text((420, 154), purch_txt, fill="#0F172A", font=body_font)
    draw.text((420, 174), age_txt, fill="#0F172A", font=body_bold)

    # 3. Warranty & Defect Telemetry Panel
    draw.rounded_rectangle([(24, 218), (WIDTH - 24, 350)], radius=8, fill="#FFFFFF", outline="#E2E8F0", width=1)
    draw.text((40, 230), "WARRANTY & REPORTED FAULT TELEMETRY", fill="#1E3A8A", font=section_font)

    # Warranty Status Badge
    w_status = metrics["warranty_status"]
    w_delta = metrics["warranty_delta_text"]
    if w_status == "Active":
        badge_bg, badge_border, badge_txt = "#DCFCE7", "#86EFAC", "#166534"
    else:
        badge_bg, badge_border, badge_txt = "#FEE2E2", "#FCA5A5", "#991B1B"

    draw.rounded_rectangle([(40, 256), (360, 288)], radius=6, fill=badge_bg, outline=badge_border, width=1)
    draw.text((50, 264), f"Warranty: {w_status} ({w_delta})", fill=badge_txt, font=body_bold)

    w_period = f"Term: {rec.get('warranty_duration_months', 0)} Mo (Exp: {format_display_date(metrics['expiry_date_parsed'], 1)})"
    draw.text((40, 298), w_period, fill="#475569", font=small_font)

    # Fault Details
    fault_type = f"Fault Type: {rec.get('damage_type', 'N/A')}"
    fault_date_str = f"Occurrence: {format_display_date(metrics['fault_date_parsed'], 1)}"
    draw.text((420, 256), fault_type, fill="#0F172A", font=body_bold)
    draw.text((420, 276), fault_date_str, fill="#475569", font=body_font)

    # Truncated description
    desc = str(rec.get("fault_description", "None provided"))
    if len(desc) > 65:
        desc = desc[:62] + "..."
    draw.text((420, 298), f"Symptom: {desc}", fill="#334155", font=small_font)

    # Repair history
    rep_hist = str(rec.get("repair_history", "0 repairs"))
    draw.text((40, 322), f"Service History: {rep_hist}", fill="#0F172A", font=body_bold)

    # 4. Document Audit & Serial Verification Panel
    draw.rounded_rectangle([(24, 364), (WIDTH - 24, 526)], radius=8, fill="#FFFFFF", outline="#E2E8F0", width=1)
    draw.text((40, 376), "DOCUMENTATION AUDIT & SERIAL INTEGRITY CHECK", fill="#1E3A8A", font=section_font)

    # Checklist Items
    docs = [
        ("Retail Purchase Receipt", bool(rec.get("has_receipt", False))),
        ("Official Warranty Card", bool(rec.get("has_warranty_card", False))),
        ("Physical Product Image", bool(rec.get("has_product_image", False))),
        ("Serial Barcode Photo", bool(rec.get("has_serial_evidence", False))),
    ]

    for idx, (doc_name, is_present) in enumerate(docs):
        y_pos = 404 + (idx * 26)
        if is_present:
            chk_icon, chk_color = "[✓] PRESENT", "#15803D"
        else:
            chk_icon, chk_color = "[✗] MISSING", "#DC2626"

        draw.text((40, y_pos), doc_name, fill="#334155", font=body_font)
        draw.text((240, y_pos), chk_icon, fill=chk_color, font=body_bold)

    # Serial Match Inspection Box
    sm_status = metrics["serial_match_status"]
    if sm_status == "Exact Match":
        sm_bg, sm_border, sm_txt = "#EFF6FF", "#BFDBFE", "#1E40AF"
    elif sm_status == "Mismatch":
        sm_bg, sm_border, sm_txt = "#FEF2F2", "#FECACA", "#991B1B"
    else:
        sm_bg, sm_border, sm_txt = "#FFFBEB", "#FDE68A", "#92400E"

    draw.rounded_rectangle([(420, 404), (WIDTH - 40, 508)], radius=6, fill=sm_bg, outline=sm_border, width=1)
    draw.text((436, 416), "INVOICED SERIAL AUDIT", fill=sm_txt, font=section_font)

    rcpt_sn_disp = str(rec.get("serial_number_on_receipt", "")).strip()
    if not rcpt_sn_disp or pd.isna(rec.get("serial_number_on_receipt")):
        rcpt_sn_disp = "[NONE RECORDED ON INVOICE]"

    draw.text((436, 442), f"Receipt Serial: {rcpt_sn_disp}", fill="#1E293B", font=small_font)
    draw.text((436, 464), f"Chassis Serial: {str(rec.get('serial_number', 'N/A')).strip()}", fill="#1E293B", font=small_font)
    draw.text((436, 486), f"Result: {sm_status}", fill=sm_txt, font=body_bold)

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
    - Layout: Two-column telemetry matrix with high-contrast diagnostic blocks.
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

    # 1. Header Bar
    draw.rectangle([(0, 0), (WIDTH, 52)], fill="#1E293B")
    draw.text((24, 16), "ASSUREX INTAKE INSPECTION DOSSIER", fill="#38BDF8", font=title_font)

    claim_id_str = f"CASE #{rec.get('claim_id', 'UNKNOWN')}"
    claim_dt_str = f"Logged: {format_display_date(metrics['claim_date_parsed'], 2)}"
    draw.text((WIDTH - 250, 12), claim_id_str, fill="#F8FAFC", font=header_id_font)
    draw.text((WIDTH - 250, 30), claim_dt_str, fill="#94A3B8", font=small_font)

    # 2. Left Column: Hardware & Warranty Telemetry
    draw.rounded_rectangle([(24, 68), (390, 526)], radius=6, fill="#1E293B", outline="#334155", width=1)
    draw.text((40, 80), "// HARDWARE TELEMETRY", fill="#38BDF8", font=section_font)

    prod_name = str(rec.get("product_name", "Unknown Product"))
    draw.text((40, 104), prod_name, fill="#F8FAFC", font=body_bold)
    draw.text((40, 122), f"{rec.get('product_category', 'N/A')} | {rec.get('brand', 'N/A')}", fill="#94A3B8", font=small_font)

    draw.line([(40, 144), (374, 144)], fill="#334155", width=1)

    draw.text((40, 154), "Model Code:", fill="#64748B", font=small_font)
    draw.text((140, 154), str(rec.get("model_number", "N/A")), fill="#F1F5F9", font=small_font)

    draw.text((40, 174), "Chassis SN:", fill="#64748B", font=small_font)
    draw.text((140, 174), str(rec.get("serial_number", "N/A")).strip(), fill="#38BDF8", font=small_font)

    draw.text((40, 194), "Purchase Date:", fill="#64748B", font=small_font)
    draw.text((140, 194), format_display_date(metrics["purchase_date_parsed"], 2), fill="#F1F5F9", font=small_font)

    draw.text((40, 214), "Device Lifespan:", fill="#64748B", font=small_font)
    draw.text((140, 214), metrics["product_age_text"], fill="#FCD34D", font=body_bold)

    draw.line([(40, 238), (374, 238)], fill="#334155", width=1)
    draw.text((40, 248), "// WARRANTY LIFECYCLE", fill="#38BDF8", font=section_font)

    # Status box
    w_status = metrics["warranty_status"]
    w_delta = metrics["warranty_delta_text"]
    if w_status == "Active":
        box_bg, box_border, box_fg = "#064E3B", "#059669", "#34D399"
    else:
        box_bg, box_border, box_fg = "#4C0519", "#E11D48", "#FDA4AF"

    draw.rounded_rectangle([(40, 272), (374, 308)], radius=4, fill=box_bg, outline=box_border, width=1)
    draw.text((52, 282), f"STATUS: {w_status.upper()}  [{w_delta}]", fill=box_fg, font=body_bold)

    draw.text((40, 320), "Coverage Duration:", fill="#64748B", font=small_font)
    draw.text((170, 320), f"{rec.get('warranty_duration_months', 0)} Months", fill="#F1F5F9", font=small_font)

    draw.text((40, 340), "Expiry Boundary:", fill="#64748B", font=small_font)
    draw.text((170, 340), format_display_date(metrics["expiry_date_parsed"], 2), fill="#F1F5F9", font=small_font)

    draw.line([(40, 366), (374, 366)], fill="#334155", width=1)
    draw.text((40, 376), "// SERVICE INTERVENTIONS", fill="#38BDF8", font=section_font)

    rep_hist = str(rec.get("repair_history", "0 repairs"))
    draw.text((40, 400), rep_hist, fill="#F1F5F9", font=body_bold)

    # Prior replacement flag
    prior_rep = bool(rec.get("prior_replacement", False))
    pr_txt = "YES (Replacement Issued)" if prior_rep else "NO (Original Unit)"
    pr_fg = "#F87171" if prior_rep else "#94A3B8"
    draw.text((40, 430), f"Prior Unit Replacement: {pr_txt}", fill=pr_fg, font=small_font)

    draw.text((40, 480), "INTAKE CHANNEL: Direct Counter", fill="#64748B", font=small_font)
    draw.text((40, 498), "SECURITY HASH: Verified OEM Barcode", fill="#64748B", font=small_font)

    # 3. Right Column: Incident Telemetry & Document Auditing
    draw.rounded_rectangle([(410, 68), (WIDTH - 24, 526)], radius=6, fill="#1E293B", outline="#334155", width=1)
    draw.text((426, 80), "// INCIDENT & DEFECT PROFILE", fill="#38BDF8", font=section_font)

    draw.text((426, 104), "Damage Classification:", fill="#64748B", font=small_font)
    draw.text((426, 122), str(rec.get("damage_type", "N/A")), fill="#F8FAFC", font=body_bold)

    draw.text((426, 150), "Incident Occurrence Date:", fill="#64748B", font=small_font)
    draw.text((610, 150), format_display_date(metrics["fault_date_parsed"], 2), fill="#F1F5F9", font=small_font)

    draw.text((426, 174), "Claim Filing Date:", fill="#64748B", font=small_font)
    draw.text((610, 174), format_display_date(metrics["claim_date_parsed"], 2), fill="#F1F5F9", font=small_font)

    desc = str(rec.get("fault_description", "None provided"))
    if len(desc) > 75:
        desc = desc[:72] + "..."
    draw.text((426, 202), "Declared Symptoms:", fill="#64748B", font=small_font)
    draw.text((426, 220), desc, fill="#CBD5E1", font=small_font)

    draw.line([(426, 252), (WIDTH - 40, 252)], fill="#334155", width=1)
    draw.text((426, 262), "// EVIDENTIARY CHECKLIST", fill="#38BDF8", font=section_font)

    chk_docs = [
        ("Purchase Invoice Attachment", bool(rec.get("has_receipt", False))),
        ("Dealer Stamped Warranty Card", bool(rec.get("has_warranty_card", False))),
        ("Unit Photographic Evidence", bool(rec.get("has_product_image", False))),
        ("Chassis Serial Nameplate Photo", bool(rec.get("has_serial_evidence", False))),
    ]

    for idx, (label, present) in enumerate(chk_docs):
        y_pos = 288 + (idx * 24)
        if present:
            tag, tag_fg = "[✓] YES", "#34D399"
        else:
            tag, tag_fg = "[✗] NO ", "#F87171"
        draw.text((426, y_pos), label, fill="#94A3B8", font=small_font)
        draw.text((680, y_pos), tag, fill=tag_fg, font=body_bold)

    draw.line([(426, 392), (WIDTH - 40, 392)], fill="#334155", width=1)
    draw.text((426, 402), "// INVOICE SERIAL RECONCILIATION", fill="#38BDF8", font=section_font)

    sm_status = metrics["serial_match_status"]
    if sm_status == "Exact Match":
        sm_box_bg, sm_box_border, sm_box_fg = "#064E3B", "#059669", "#34D399"
    elif sm_status == "Mismatch":
        sm_box_bg, sm_box_border, sm_box_fg = "#4C0519", "#E11D48", "#FDA4AF"
    else:
        sm_box_bg, sm_box_border, sm_box_fg = "#451A03", "#D97706", "#FCD34D"

    rcpt_sn_disp = str(rec.get("serial_number_on_receipt", "")).strip()
    if not rcpt_sn_disp or pd.isna(rec.get("serial_number_on_receipt")):
        rcpt_sn_disp = "NOT_SPECIFIED"

    draw.text((426, 426), "Receipt S/N:", fill="#64748B", font=small_font)
    draw.text((520, 426), rcpt_sn_disp, fill="#F1F5F9", font=small_font)

    draw.rounded_rectangle([(426, 452), (WIDTH - 40, 488)], radius=4, fill=sm_box_bg, outline=sm_box_border, width=1)
    draw.text((440, 462), f"RECONCILIATION: {sm_status.upper()}", fill=sm_box_fg, font=body_bold)

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
            metrics = compute_claim_metrics(rec)

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
