"""
AssureX Unified Claim Metrics Engine
====================================

SINGLE SOURCE OF TRUTH for all date-derived warranty claim calculations:
1. `product_age_days`: Elapsed calendar days from purchase_date to claim_filing_date.
2. `warranty_expiry_date`: Contractual expiry date using exact calendar-month
   arithmetic via `dateutil.relativedelta` (NOT 30-day approximations).
3. `remaining_warranty_days`: Coverage days remaining from claim_filing_date to expiry_date.
4. `warranty_status`: "Active" if remaining_warranty_days >= 0 else "Expired".

This module is the ONLY place these values are ever calculated anywhere in the project.
"""

from datetime import datetime, date
from typing import Union, Dict, Any, Optional
from dateutil.relativedelta import relativedelta


def parse_date(val: Any) -> date:
    """Robustly parses a string, datetime, or date into a datetime.date object."""
    if val is None:
        raise ValueError("Cannot parse None as date.")
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    clean_str = str(val).strip().split("T")[0]
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(clean_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date string: '{val}'")


class ClaimMetrics(dict):
    """
    Structured dictionary containing the 4 canonical claim metrics.
    Supports dictionary indexing (`res['product_age_days']`), attribute
    access (`res.product_age_days`), and tuple unpacking.
    """

    def __init__(
        self,
        product_age_days: int,
        warranty_expiry_date: str,
        remaining_warranty_days: int,
        warranty_status: str,
        warranty_expiry_date_obj: date
    ):
        super().__init__(
            product_age_days=product_age_days,
            warranty_expiry_date=warranty_expiry_date,
            remaining_warranty_days=remaining_warranty_days,
            warranty_status=warranty_status
        )
        self._expiry_date_obj = warranty_expiry_date_obj

    @property
    def product_age_days(self) -> int:
        return self["product_age_days"]

    @property
    def warranty_expiry_date(self) -> str:
        return self["warranty_expiry_date"]

    @property
    def remaining_warranty_days(self) -> int:
        return self["remaining_warranty_days"]

    @property
    def warranty_status(self) -> str:
        return self["warranty_status"]

    @property
    def expiry_date_obj(self) -> date:
        return self._expiry_date_obj

    def __iter__(self):
        """Allows unpacking: age, exp, rem, status = compute_claim_metrics(...)"""
        yield self["product_age_days"]
        yield self["warranty_expiry_date"]
        yield self["remaining_warranty_days"]
        yield self["warranty_status"]


def compute_claim_metrics(
    purchase_date: Union[str, date, datetime],
    claim_filing_date: Union[str, date, datetime],
    warranty_duration_months: Union[int, float, str]
) -> ClaimMetrics:
    """
    Computes canonical date metrics for an AssureX warranty claim.

    Parameters:
    -----------
    purchase_date : Union[str, date, datetime]
        The date the product was purchased by the consumer.
    claim_filing_date : Union[str, date, datetime]
        The date the warranty claim was officially submitted / filed.
    warranty_duration_months : Union[int, float, str]
        Contractual warranty duration in months (e.g. 12, 24, 36).

    Returns:
    --------
    ClaimMetrics
        {
            "product_age_days": int,            # (claim_filing_date - purchase_date).days
            "warranty_expiry_date": str,        # YYYY-MM-DD via relativedelta(months=...)
            "remaining_warranty_days": int,     # (warranty_expiry_date - claim_filing_date).days
            "warranty_status": str              # "Active" if remaining_warranty_days >= 0 else "Expired"
        }
    """
    p_date = parse_date(purchase_date)
    c_date = parse_date(claim_filing_date)
    duration = int(warranty_duration_months)

    # 1. Product age at claim filing (elapsed calendar days)
    product_age_days = (c_date - p_date).days

    # 2. Expiry date computed with exact calendar-month arithmetic
    expiry_date_obj = p_date + relativedelta(months=duration)
    warranty_expiry_date_str = expiry_date_obj.strftime("%Y-%m-%d")

    # 3. Remaining warranty days at claim filing
    remaining_warranty_days = (expiry_date_obj - c_date).days

    # 4. Canonical warranty status
    warranty_status = "Active" if remaining_warranty_days >= 0 else "Expired"

    return ClaimMetrics(
        product_age_days=product_age_days,
        warranty_expiry_date=warranty_expiry_date_str,
        remaining_warranty_days=remaining_warranty_days,
        warranty_status=warranty_status,
        warranty_expiry_date_obj=expiry_date_obj
    )


# Also expose in src for library packaging
if __name__ == "__main__":
    print("Testing claim_metrics.py...")
    res = compute_claim_metrics("2025-01-15", "2025-07-20", 12)
    print("Output dict:", dict(res))
    print(f"Age: {res.product_age_days}, Expiry: {res.warranty_expiry_date}, Remaining: {res.remaining_warranty_days}, Status: {res.warranty_status}")
    age, exp, rem, stat = res
    print(f"Unpacked: age={age}, exp={exp}, rem={rem}, stat={stat}")
    print("[PASS] claim_metrics test completed successfully.")
