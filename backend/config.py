"""
AssureX Backend Configuration
=============================
Central configuration settings for database, security, and storage paths.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

UPLOADS_DIR = BASE_DIR / "uploads" / "documents"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

POLICIES_DIR = BASE_DIR / "policies"

# Database Configuration (SQLite default, PostgreSQL compatible)
DEFAULT_DB_PATH = DATABASE_DIR / "assurex.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# JWT Security Settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "assurex-super-secret-jwt-key-2026-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours

# Role Definitions
ROLE_CUSTOMER = "customer"
ROLE_SERVICE_CENTER = "service_center"
ROLE_CLAIM_REVIEWER = "claim_reviewer"
ROLE_ADMIN = "admin"

ALL_ROLES = [ROLE_CUSTOMER, ROLE_SERVICE_CENTER, ROLE_CLAIM_REVIEWER, ROLE_ADMIN]
