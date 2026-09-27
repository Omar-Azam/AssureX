"""
AssureX PDF Claim Adjudication Report Generator
===============================================
Generates official, downloadable multi-page PDF evaluation reports using ReportLab.
"""

import io
import json
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

from backend.models import Claim


def generate_claim_pdf_report(claim: Claim) -> io.BytesIO:
    """
    Builds a professional, comprehensive PDF evaluation report for a given Claim.
    Returns in-memory BytesIO buffer ready for streaming HTTP download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=0
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b")
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6
    )
    cell_style = ParagraphStyle(
        "CellNormal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155")
    )
    cell_bold = ParagraphStyle(
        "CellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("AssureX Warranty Claims Engine", title_style))
    story.append(Paragraph(f"Official Claim Adjudication & Evaluation Report • Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2563eb"), spaceAfter=14))

    # 2. Claim Summary Table
    product = claim.product
    warranty = product.warranty if product else None
    pred = claim.prediction

    final_decision = pred.final_decision if pred else claim.status
    consistency = pred.model_consistency_status if pred else "N/A"

    # Color code decision banner
    decision_color = colors.HexColor("#0284c7")  # Blue default
    if "valid" in final_decision.lower() and "invalid" not in final_decision.lower():
        decision_color = colors.HexColor("#16a34a")  # Green
    elif "invalid" in final_decision.lower():
        decision_color = colors.HexColor("#dc2626")  # Red
    elif "manual" in final_decision.lower() or "review" in final_decision.lower():
        decision_color = colors.HexColor("#d97706")  # Amber

    decision_banner_data = [
        [
            Paragraph(f"<b>FINAL ADJUDICATION DECISION:</b> {final_decision.upper()}", ParagraphStyle("Dec", fontName="Helvetica-Bold", fontSize=13, textColor=colors.white)),
            Paragraph(f"<b>Status:</b> {claim.status}", ParagraphStyle("Stat", fontName="Helvetica-Bold", fontSize=11, textColor=colors.white, alignment=2))
        ]
    ]
    t_banner = Table(decision_banner_data, colWidths=[360, 180])
    t_banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), decision_color),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    story.append(t_banner)
    story.append(Spacer(1, 12))

    # 3. Claim & Product Metadata Table
    story.append(Paragraph("1. Claim & Product Identification", section_heading))
    meta_data = [
        [Paragraph("Claim ID", cell_bold), Paragraph(str(claim.claim_id), cell_style),
         Paragraph("Submission Date", cell_bold), Paragraph(str(claim.claim_submission_date), cell_style)],
        [Paragraph("Product Name", cell_bold), Paragraph(str(product.product_name if product else "N/A"), cell_style),
         Paragraph("Category", cell_bold), Paragraph(str(product.product_category if product else "N/A"), cell_style)],
        [Paragraph("Brand & Model", cell_bold), Paragraph(f"{product.brand if product else ''} ({product.model_number if product else ''})", cell_style),
         Paragraph("Serial Number", cell_bold), Paragraph(str(product.serial_number if product else "N/A"), cell_style)],
        [Paragraph("Purchase Date", cell_bold), Paragraph(str(product.purchase_date if product else "N/A"), cell_style),
         Paragraph("Retailer", cell_bold), Paragraph(str(product.retailer if product else "N/A"), cell_style)],
        [Paragraph("Warranty Status", cell_bold), Paragraph(str(warranty.warranty_status if warranty else "N/A"), cell_style),
         Paragraph("Warranty Expiry", cell_bold), Paragraph(str(warranty.warranty_expiry_date if warranty else "N/A"), cell_style)],
        [Paragraph("Damage Type", cell_bold), Paragraph(str(claim.damage_type), cell_style),
         Paragraph("Fault Occurrence", cell_bold), Paragraph(str(claim.fault_occurrence_date), cell_style)],
    ]
    t_meta = Table(meta_data, colWidths=[110, 160, 110, 160])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # 4. Fault Description
    story.append(Paragraph("2. Reported Fault Narrative", section_heading))
    fault_table = Table([[Paragraph(str(claim.fault_description), cell_style)]], colWidths=[540])
    fault_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(fault_table)
    story.append(Spacer(1, 10))

    # 5. Multimodal AI & Rule Engine Analysis
    story.append(Paragraph("3. Multimodal Decision Engine Evaluation", section_heading))
    if pred:
        try:
            py_res = json.loads(pred.python_prediction) if isinstance(pred.python_prediction, str) else pred.python_prediction
            gtm_res = json.loads(pred.gtm_prediction) if isinstance(pred.gtm_prediction, str) else pred.gtm_prediction
            rule_res = json.loads(pred.rule_engine_result) if isinstance(pred.rule_engine_result, str) else pred.rule_engine_result
            dec_res = json.loads(pred.decision_engine_result) if isinstance(pred.decision_engine_result, str) else pred.decision_engine_result
        except Exception:
            py_res, gtm_res, rule_res, dec_res = {}, {}, {}, {}

        expl = dec_res.get("decision_explanation", {})

        ai_data = [
            [Paragraph("Evaluation Dimension", cell_bold), Paragraph("Model Outcome", cell_bold), Paragraph("Confidence / Detail", cell_bold)],
            [Paragraph("Tabular Model (Python)", cell_style),
             Paragraph(str(py_res.get("predicted_class", "N/A")), cell_style),
             Paragraph(f"Valid: {py_res.get('confidence_valid', 0)*100:.1f}% | Invalid: {py_res.get('confidence_invalid', 0)*100:.1f}%", cell_style)],
            [Paragraph("Visual Card (GTM)", cell_style),
             Paragraph(str(gtm_res.get("predicted_class", "N/A")), cell_style),
             Paragraph(f"Valid: {gtm_res.get('confidence_valid', 0)*100:.1f}% | Invalid: {gtm_res.get('confidence_invalid', 0)*100:.1f}%", cell_style)],
            [Paragraph("Model Consistency", cell_style),
             Paragraph(str(consistency), cell_style),
             Paragraph(f"Confidence Gap: {pred.confidence_difference*100:.1f}%", cell_style)],
            [Paragraph("Rule Engine Checks", cell_style),
             Paragraph(f"Passed: {len(rule_res.get('rules_passed', []))} | Failed: {len(rule_res.get('rules_failed', []))}", cell_style),
             Paragraph(f"Manual Review Triggered: {rule_res.get('manual_review_required', False)}", cell_style)],
        ]
        t_ai = Table(ai_data, colWidths=[160, 160, 220])
        t_ai.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t_ai)
        story.append(Spacer(1, 10))

        # Contradictions Section
        contradictions = expl.get("contradictions", []) or rule_res.get("evaluations", {}).get("contradictions", []) or rule_res.get("contradictions", [])
        if contradictions:
            story.append(Paragraph("<b><font color='#dc2626'>Integrity & Temporal Contradictions Flagged:</font></b>", cell_bold))
            for c in contradictions:
                story.append(Paragraph(f"⚠️ {c}", ParagraphStyle("Contra", parent=cell_style, textColor=colors.HexColor("#dc2626"))))
            story.append(Spacer(1, 4))

        # Final Recommendation & Summary
        summary_text = expl.get("summary") or f"Final automated recommendation: {final_decision}. Consistency status: {consistency}."
        story.append(Paragraph("<b>Final Recommendation Rationale:</b>", cell_bold))
        story.append(Paragraph(str(summary_text), cell_style))
        story.append(Spacer(1, 6))

        # Factors Supporting / Opposing
        factors_sup = expl.get("factors_supporting", [])
        factors_opp = expl.get("factors_opposing", [])
        evidence_needed = expl.get("additional_evidence_needed", []) or expl.get("evidence_needed", [])

        if factors_sup:
            story.append(Paragraph("<b>Factors Supporting Decision:</b>", cell_bold))
            for f in factors_sup[:4]:
                story.append(Paragraph(f"• {f}", cell_style))
            story.append(Spacer(1, 4))

        if factors_opp:
            story.append(Paragraph("<b>Factors Opposing / Caveats:</b>", cell_bold))
            for f in factors_opp[:3]:
                story.append(Paragraph(f"• {f}", cell_style))
            story.append(Spacer(1, 4))

        if evidence_needed:
            story.append(Paragraph("<b>Additional Evidence Required:</b>", cell_bold))
            for e in evidence_needed[:4]:
                story.append(Paragraph(f"• {e}", cell_style))
            story.append(Spacer(1, 6))

    # 6. Supporting Documents Attached
    story.append(Paragraph("4. Supporting Evidence & Verification Audit", section_heading))
    doc_rows = [[Paragraph("File Name", cell_bold), Paragraph("Type", cell_bold), Paragraph("SHA-256 Hash", cell_bold), Paragraph("Duplicate?", cell_bold)]]
    for d in claim.documents:
        doc_rows.append([
            Paragraph(str(d.file_name), cell_style),
            Paragraph(str(d.file_type), cell_style),
            Paragraph(str(d.sha256_hash[:16]) + "...", cell_style),
            Paragraph("YES (Flagged)" if d.is_duplicate else "Clean", cell_bold if d.is_duplicate else cell_style)
        ])
    if len(doc_rows) == 1:
        doc_rows.append([Paragraph("No documents uploaded", cell_style), Paragraph("-", cell_style), Paragraph("-", cell_style), Paragraph("-", cell_style)])

    t_docs = Table(doc_rows, colWidths=[160, 100, 180, 100])
    t_docs.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_docs)
    story.append(Spacer(1, 14))

    # 7. Reviewer Sign-off & Adjudication Notes Box
    story.append(Paragraph("5. Adjudicator Review & Authorization", section_heading))
    
    if claim.reviews:
        rev_rows = [[Paragraph("Reviewer", cell_bold), Paragraph("Action", cell_bold), Paragraph("Date", cell_bold), Paragraph("Decision Notes / Reviewer Comments", cell_bold)]]
        for r in claim.reviews:
            reviewer_name = r.reviewer.full_name if r.reviewer else f"Reviewer #{r.reviewer_id}"
            rev_rows.append([
                Paragraph(reviewer_name, cell_bold),
                Paragraph(str(r.action).upper(), cell_style),
                Paragraph(r.created_at.strftime('%Y-%m-%d %H:%M'), cell_style),
                Paragraph(str(r.decision_notes) + (f" [Override: {r.overridden_decision}]" if r.overridden_decision else ""), cell_style)
            ])
        t_rev = Table(rev_rows, colWidths=[110, 80, 100, 250])
        t_rev.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1"))
        ]))
        story.append(t_rev)
    else:
        rev_data = [
            [Paragraph("Reviewed By / Role:", cell_bold), Paragraph("Pending Review / Auto-Adjudicated", cell_style),
             Paragraph("Signature / Stamp:", cell_bold), Paragraph("___________________________", cell_style)],
            [Paragraph("Current Status:", cell_bold), Paragraph(str(claim.status), cell_style),
             Paragraph("Date of Action:", cell_bold), Paragraph(str(claim.updated_at.strftime('%Y-%m-%d')), cell_style)]
        ]
        t_rev = Table(rev_data, colWidths=[120, 150, 120, 150])
        t_rev.setStyle(TableStyle([
            ("PADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0"))
        ]))
        story.append(t_rev)

    # Build PDF document
    doc.build(story)
    buffer.seek(0)
    return buffer
