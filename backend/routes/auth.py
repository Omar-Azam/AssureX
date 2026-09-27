"""
AssureX Authentication Endpoints
================================
User registration, JWT login, and profile discovery.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.models import User, AuditLog
from backend.schemas import UserCreate, UserLogin, UserResponse, UserProfileUpdate, TokenResponse
from backend.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user
)
from backend.config import ALL_ROLES

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_in: UserCreate, request: Request, db: Session = Depends(get_db)):
    """
    Register a new user account with a designated role:
    - customer
    - service_center
    - claim_reviewer
    - admin
    """
    if user_in.role not in ALL_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join(ALL_ROLES)}"
        )

    # Check for existing username or email
    if db.query(User).filter(User.username == user_in.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username is already registered."
        )

    if db.query(User).filter(User.email == user_in.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email address is already registered."
        )

    new_user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
        full_name=user_in.full_name,
        role=user_in.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Audit log registration
    audit = AuditLog(
        user_id=new_user.id,
        action="USER_REGISTRATION",
        entity_type="User",
        entity_id=str(new_user.id),
        details=f"User registered with role '{new_user.role}'",
        ip_address=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    return new_user


@router.post("/login", response_model=TokenResponse)
def login_for_access_token(
    user_in: UserLogin,
    request: Request,
    db: Session = Depends(get_db)
):
    """Authenticate with username or email and password, returning JWT bearer access token."""
    login_id = user_in.username.strip()
    user = db.query(User).filter(
        (User.username == login_id) | (func.lower(User.email) == login_id.lower())
    ).first()
    if not user or not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password."
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account has been deactivated."
        )

    token = create_access_token(data={"sub": str(user.id), "role": user.role})

    # Log successful login in AuditLog
    audit = AuditLog(
        user_id=user.id,
        action="LOGIN",
        entity_type="User",
        entity_id=str(user.id),
        details=f"User '{user.username}' logged in successfully",
        ip_address=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user
    }


@router.post("/token", response_model=TokenResponse)
async def login_oauth2_token(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Standard OAuth2 token endpoint compatible with form-data and JSON bodies.
    Accepts username/email and password.
    """
    content_type = request.headers.get("content-type", "")
    username = None
    password = None

    if "application/json" in content_type:
        body = await request.json()
        username = body.get("username")
        password = body.get("password")
    else:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")

    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password are required."
        )

    login_id = str(username).strip()
    user = db.query(User).filter(
        (User.username == login_id) | (func.lower(User.email) == login_id.lower())
    ).first()

    if not user or not verify_password(str(password), user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password."
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account has been deactivated."
        )

    token = create_access_token(data={"sub": str(user.id), "role": user.role})

    audit = AuditLog(
        user_id=user.id,
        action="LOGIN",
        entity_type="User",
        entity_id=str(user.id),
        details=f"User '{user.username}' logged in via /token endpoint",
        ip_address=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user
    }


@router.get("/me", response_model=UserResponse)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Retrieve profile information for the authenticated user."""
    return current_user


@router.put("/profile", response_model=UserResponse)
def update_user_profile(
    profile_in: UserProfileUpdate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update profile information (full_name, email, password) for the authenticated user."""
    if profile_in.email and profile_in.email != current_user.email:
        existing = db.query(User).filter(User.email == profile_in.email, User.id != current_user.id).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email address is already in use by another account."
            )
        current_user.email = profile_in.email

    if profile_in.full_name:
        current_user.full_name = profile_in.full_name

    if profile_in.password:
        current_user.hashed_password = hash_password(profile_in.password)

    db.commit()
    db.refresh(current_user)

    audit = AuditLog(
        user_id=current_user.id,
        action="PROFILE_UPDATE",
        entity_type="User",
        entity_id=str(current_user.id),
        details="User updated account profile details",
        ip_address=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    return current_user
