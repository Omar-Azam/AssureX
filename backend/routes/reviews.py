"""
AssureX Manual Review Queue and Adjudication Endpoints
======================================================
Queue inspection, claim approval, rejection, and underwriter overrides.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Claim, Review, AuditLog, User
from backend.schemas import ClaimResponse, ReviewCreate, ReviewResponse
from backend.auth import get_current_user, require_roles
from backend.config import ROLE_CLAIM_REVIEWER, ROLE_ADMIN

router = APIRouter(tags=["Manual Review Queue"])


@router.get("/api/review-queue", response_model=List[ClaimResponse])
def get_manual_review_queue(
    current_user: User = Depends(require_roles([ROLE_CLAIM_REVIEWER, ROLE_ADMIN])),
    db: Session = Depends(get_db)
):
    """
    Retrieve all claims pending human manual review or adjudication.
    Restricted to claim reviewers and administrators.
    """
    return db.query(Claim).filter(
        Claim.status.in_(["Manual Review", "Under Evaluation", "Submitted"])
    ).order_by(Claim.created_at.asc()).all()


@router.post("/api/claims/{claim_id}/review", response_model=ReviewResponse)
def submit_claim_review(
    claim_id: str,
    review_in: ReviewCreate,
    request: Request,
    current_user: User = Depends(require_roles([ROLE_CLAIM_REVIEWER, ROLE_ADMIN])),
    db: Session = Depends(get_db)
):
    """
    Submit a reviewer decision or underwriter override on a claim:
    - approve: Sets claim status to 'Approved'
    - reject: Sets claim status to 'Rejected'
    - override: Sets claim status to 'Overridden' with original decision reference
    - request_evidence: Sets claim status to 'Pending Evidence'
    """
    claim = db.query(Claim).filter((Claim.claim_id == claim_id) | (Claim.id == int(claim_id) if claim_id.isdigit() else False)).first()
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target claim not found.")

    action_lower = review_in.action.strip().lower()
    valid_actions = {"approve", "reject", "override", "request_evidence"}
    if action_lower not in valid_actions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid review action. Must be one of: {', '.join(valid_actions)}"
        )

    old_status = claim.status
    if action_lower == "approve":
        claim.status = "Approved"
    elif action_lower == "reject":
        claim.status = "Rejected"
    elif action_lower == "override":
        claim.status = "Overridden"
    elif action_lower == "request_evidence":
        claim.status = "Pending Evidence"

    review = Review(
        claim_id=claim.id,
        reviewer_id=current_user.id,
        action=action_lower,
        decision_notes=review_in.decision_notes,
        overridden_decision=review_in.overridden_decision
    )
    db.add(review)
    db.commit()
    db.refresh(review)

    # Log action to AuditLog
    audit = AuditLog(
        user_id=current_user.id,
        action=f"REVIEWER_ACTION_{action_lower.upper()}",
        entity_type="Claim",
        entity_id=claim.claim_id,
        details=f"Reviewer '{current_user.username}' executed '{action_lower}'. Status changed from '{old_status}' to '{claim.status}'. Notes: {review.decision_notes[:100]}",
        ip_address=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    return review
