"""
Audit Script: Recomputed Card Metrics vs. Rendered Card Values
==============================================================

For every record in claims_dataset.csv, independently recomputes:
1. product_age_days = (claim_filing_date - purchase_date).days
2. warranty_expiry_date = purchase_date + warranty_duration_months
   (using exact calendar-month arithmetic via dateutil.relativedelta)
3. remaining_warranty_days = warranty_expiry_date (recomputed) - claim_filing_date

Compares each of these three against what is actually rendered on the
corresponding Claim Summary Card / CSV field.

Reports:
- Exact match count and mismatch count for all three metrics separately.
- Side-by-side factual comparison of actual values for mismatched records.
"""

import os
import argparse
from datetime import datetime, date
from typing import Dict, Any, List, Optional
import pandas as pd
from claim_metrics import compute_claim_metrics, parse_date


def parse_date(val: Any) -> Optional[date]:
    """Parse date from string or date object."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    try:
        clean = str(val).strip().split("T")[0]
        return datetime.strptime(clean, "%Y-%m-%d").date()
    except Exception:
        return None


def run_audit(input_csv: str = "claims_dataset.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes independent recomputations and compares against rendered card/CSV values.
    """
    if not os.path.exists(input_csv):
        # Fallback candidate paths
        candidates = [
            os.path.join("dataset", "claims_dataset.csv"),
            "dataset/train.csv",
        ]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if not found:
            raise FileNotFoundError(f"Cannot locate input dataset CSV at '{input_csv}'.")
        input_csv = found

    # If loading train.csv alone, combine all splits to get full dataset
    if input_csv.endswith("train.csv"):
        dfs = []
        for s in ["train.csv", "val.csv", "test.csv"]:
            p = os.path.join(os.path.dirname(input_csv), s)
            if os.path.exists(p):
                dfs.append(pd.read_csv(p))
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(input_csv)

    results: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        claim_id = str(row.get("claim_id", "UNKNOWN"))

        # Date parsing
        purchase_d = parse_date(row.get("purchase_date"))
        fault_d = parse_date(row.get("fault_occurrence_date"))
        filing_d = parse_date(row.get("claim_submission_date") or row.get("claim_filing_date"))
        rendered_expiry_d = parse_date(row.get("warranty_expiry_date"))
        duration_months = int(row.get("warranty_duration_months", 12))

        # ---------------------------------------------------------------------
        # Canonical Recomputations via single source of truth (claim_metrics.py)
        # ---------------------------------------------------------------------
        if purchase_d and filing_d:
            c_metrics = compute_claim_metrics(purchase_d, filing_d, duration_months)
            recomp_product_age = c_metrics.product_age_days
            recomp_expiry_d = c_metrics.expiry_date_obj
            recomp_remaining_days = c_metrics.remaining_warranty_days
        else:
            recomp_product_age = None
            recomp_expiry_d = None
            recomp_remaining_days = None

        # ---------------------------------------------------------------------
        # Rendered Values on Card / Corrected CSV
        # ---------------------------------------------------------------------
        from generate_claim_cards import prepare_claim_card_metrics
        card_tel = prepare_claim_card_metrics(row.to_dict())
        rendered_product_age = card_tel["product_age_days"]
        rendered_expiry = rendered_expiry_d
        rendered_remaining_days = card_tel["remaining_warranty_days"]

        # ---------------------------------------------------------------------
        # Equality Comparisons
        # ---------------------------------------------------------------------
        match_age = (recomp_product_age == rendered_product_age)
        diff_age = (recomp_product_age - rendered_product_age) if (recomp_product_age is not None and rendered_product_age is not None) else None

        match_expiry = (recomp_expiry_d == rendered_expiry)
        diff_expiry = (recomp_expiry_d - rendered_expiry).days if (recomp_expiry_d is not None and rendered_expiry is not None) else None

        match_remaining = (recomp_remaining_days == rendered_remaining_days)
        diff_remaining = (recomp_remaining_days - rendered_remaining_days) if (recomp_remaining_days is not None and rendered_remaining_days is not None) else None

        has_any_mismatch = (not match_age) or (not match_expiry) or (not match_remaining)

        results.append({
            "claim_id": claim_id,
            "purchase_date": purchase_d.isoformat() if purchase_d else "N/A",
            "claim_filing_date": filing_d.isoformat() if filing_d else "N/A",
            "fault_occurrence_date": fault_d.isoformat() if fault_d else "N/A",
            "warranty_duration_months": duration_months,
            # Metric 1: Product Age
            "recomputed_product_age": recomp_product_age,
            "rendered_product_age": rendered_product_age,
            "match_product_age": match_age,
            "diff_product_age": diff_age,
            # Metric 2: Expiry Date
            "recomputed_expiry_date": recomp_expiry_d.isoformat() if recomp_expiry_d else "N/A",
            "rendered_expiry_date": rendered_expiry.isoformat() if rendered_expiry else "N/A",
            "match_expiry_date": match_expiry,
            "diff_expiry_days": diff_expiry,
            # Metric 3: Remaining Warranty Days
            "recomputed_remaining_days": recomp_remaining_days,
            "rendered_remaining_days": rendered_remaining_days,
            "match_remaining_days": match_remaining,
            "diff_remaining_days": diff_remaining,
            # Summary flag
            "has_any_mismatch": has_any_mismatch
        })

    audit_df = pd.DataFrame(results)
    total_records = len(audit_df)

    summary = {
        "total_records": total_records,
        # Metric 1: Product Age
        "product_age_exact_matches": int(audit_df["match_product_age"].sum()),
        "product_age_mismatches": int((~audit_df["match_product_age"]).sum()),
        "product_age_match_pct": round((audit_df["match_product_age"].sum() / total_records) * 100.0, 2),
        "product_age_mismatch_pct": round(((~audit_df["match_product_age"]).sum() / total_records) * 100.0, 2),
        # Metric 2: Expiry Date
        "expiry_date_exact_matches": int(audit_df["match_expiry_date"].sum()),
        "expiry_date_mismatches": int((~audit_df["match_expiry_date"]).sum()),
        "expiry_date_match_pct": round((audit_df["match_expiry_date"].sum() / total_records) * 100.0, 2),
        "expiry_date_mismatch_pct": round(((~audit_df["match_expiry_date"]).sum() / total_records) * 100.0, 2),
        # Metric 3: Remaining Warranty Days
        "remaining_days_exact_matches": int(audit_df["match_remaining_days"].sum()),
        "remaining_days_mismatches": int((~audit_df["match_remaining_days"]).sum()),
        "remaining_days_match_pct": round((audit_df["match_remaining_days"].sum() / total_records) * 100.0, 2),
        "remaining_days_mismatch_pct": round(((~audit_df["match_remaining_days"]).sum() / total_records) * 100.0, 2),
        # Overall
        "records_with_any_mismatch": int(audit_df["has_any_mismatch"].sum()),
        "records_with_all_match": int((~audit_df["has_any_mismatch"]).sum()),
    }

    return audit_df, summary


def print_report(audit_df: pd.DataFrame, summary: Dict[str, Any], limit: int = 25, show_all: bool = False) -> None:
    """Prints the factual audit summary and side-by-side values."""
    print("=" * 95)
    print("AUDIT REPORT: RECOMPUTED METRICS VS. RENDERED CARD / CSV VALUES")
    print(f"Total Records Evaluated: {summary['total_records']:,}")
    print("=" * 95)

    print("\n1. SUMMARY COUNTS BY METRIC")
    print("-" * 95)
    summary_table = pd.DataFrame([
        {
            "Metric": "1. product_age_days",
            "Formula Evaluated": "(claim_filing_date - purchase_date).days",
            "Exact Matches": f"{summary['product_age_exact_matches']:,}",
            "Match %": f"{summary['product_age_match_pct']:.2f}%",
            "Mismatches": f"{summary['product_age_mismatches']:,}",
            "Mismatch %": f"{summary['product_age_mismatch_pct']:.2f}%"
        },
        {
            "Metric": "2. warranty_expiry_date",
            "Formula Evaluated": "purchase_date + relativedelta(months=duration)",
            "Exact Matches": f"{summary['expiry_date_exact_matches']:,}",
            "Match %": f"{summary['expiry_date_match_pct']:.2f}%",
            "Mismatches": f"{summary['expiry_date_mismatches']:,}",
            "Mismatch %": f"{summary['expiry_date_mismatch_pct']:.2f}%"
        },
        {
            "Metric": "3. remaining_warranty_days",
            "Formula Evaluated": "recomputed_expiry_date - claim_filing_date",
            "Exact Matches": f"{summary['remaining_days_exact_matches']:,}",
            "Match %": f"{summary['remaining_days_match_pct']:.2f}%",
            "Mismatches": f"{summary['remaining_days_mismatches']:,}",
            "Mismatch %": f"{summary['remaining_days_mismatch_pct']:.2f}%"
        }
    ])
    print(summary_table.to_string(index=False))

    print("\n" + "-" * 95)
    print(f"Records with ALL 3 metrics matching: {summary['records_with_all_match']:,} ({summary['records_with_all_match']/summary['total_records']*100:.2f}%)")
    print(f"Records with ANY metric mismatching: {summary['records_with_any_mismatch']:,} ({summary['records_with_any_mismatch']/summary['total_records']*100:.2f}%)")

    # Mismatch Details Side-by-Side
    mismatched_df = audit_df[audit_df["has_any_mismatch"]].copy()

    display_cols = [
        "claim_id",
        "recomputed_product_age", "rendered_product_age", "diff_product_age",
        "recomputed_expiry_date", "rendered_expiry_date", "diff_expiry_days",
        "recomputed_remaining_days", "rendered_remaining_days", "diff_remaining_days"
    ]

    rename_map = {
        "claim_id": "Claim ID",
        "recomputed_product_age": "Age (Recomp)",
        "rendered_product_age": "Age (Card)",
        "diff_product_age": "Diff Age",
        "recomputed_expiry_date": "Exp (Recomp)",
        "rendered_expiry_date": "Exp (Card)",
        "diff_expiry_days": "Diff Exp (d)",
        "recomputed_remaining_days": "Rem (Recomp)",
        "rendered_remaining_days": "Rem (Card)",
        "diff_remaining_days": "Diff Rem (d)"
    }

    print("\n2. FACTUAL SIDE-BY-SIDE VALUES FOR MISMATCHED RECORDS")
    print("-" * 95)
    if show_all:
        rows_to_show = mismatched_df[display_cols].rename(columns=rename_map)
        print(f"Showing all {len(rows_to_show):,} mismatched records:")
    else:
        rows_to_show = mismatched_df[display_cols].head(limit).rename(columns=rename_map)
        print(f"Showing first {min(limit, len(mismatched_df))} of {len(mismatched_df):,} mismatched records (use --show-all to display all):")

    print(rows_to_show.to_string(index=False))
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser(description="Audit recomputed metrics vs. rendered card/CSV values")
    parser.add_argument("--input-csv", type=str, default="claims_dataset.csv", help="Path to input claims CSV")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional output path to export full side-by-side CSV")
    parser.add_argument("--limit", type=int, default=25, help="Number of mismatch rows to display in console")
    parser.add_argument("--show-all", action="store_true", help="Print all mismatched rows in console")
    args = parser.parse_args()

    audit_df, summary = run_audit(args.input_csv)
    print_report(audit_df, summary, limit=args.limit, show_all=args.show_all)

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        audit_df.to_csv(args.output_csv, index=False)
        print(f"\n[+] Full side-by-side audit export saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
