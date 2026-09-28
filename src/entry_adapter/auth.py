# PART: entry-adapter v0.2.0 (parts@637ed37)
"""Authenticate entry callers and resolve trusted scope.

Client headers are never read for scope. Scope comes only from verified request
state (or the injected ``app.state.verify_scope`` seam), with a token-gated STG
default fallback for authenticated entry calls.
"""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request
from framework.schemas.trust_level import TrustLevel


VERIFIED_TENANT_STATE_ATTR = "production_company_id"
VERIFIED_ORG_STATE_ATTR = "project_id"
STG_DEFAULT_TENANT_ENV = "STG_DEFAULT_COMPANY_ID"
STG_DEFAULT_ORG_ENV = "STG_DEFAULT_PROJECT_ID"


def authenticate_scope(request: Request) -> tuple[TrustLevel, str, str]:
    """Authenticate the caller and return its verified trust and scope."""
    trust, token_authenticated = _resolve_trust(request)
    _require_verified_trust(trust)
    tenant_id, org_id = _require_scope(request, token_authenticated=token_authenticated)
    return trust, tenant_id, org_id


def _coerce_trust(value) -> TrustLevel:
    """Normalize a request.state.trust_level to a TrustLevel enum.

    Middleware may stamp the attribute as the enum OR as its raw string value
    ("ANONYMOUS"). Comparing by identity (`is TrustLevel.ANONYMOUS`) then fails
    for the string form and silently skips token promotion / the unauthenticated
    guard. Normalize by VALUE so both forms behave identically; anything
    unrecognized falls back to the secure default (ANONYMOUS).
    """
    if isinstance(value, TrustLevel):
        return value
    try:
        return TrustLevel(value)
    except (ValueError, KeyError, TypeError):
        return TrustLevel.ANONYMOUS


def _resolve_trust(request: Request) -> tuple[TrustLevel, bool]:
    """Resolve verified caller trust and whether entry-token auth promoted it."""
    trust = _coerce_trust(getattr(request.state, "trust_level", TrustLevel.ANONYMOUS))

    # S-3 entry-point exception: INVOKE_AUTH_TOKEN is read from os.environ only
    # here because it authenticates the caller before any InvocationContext exists.
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    token_authenticated = False
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL
        token_authenticated = True
    return trust, token_authenticated


def _require_verified_trust(trust: TrustLevel) -> None:
    """Reject callers without externally verified or internal trust."""
    if trust not in {TrustLevel.VERIFIED_EXTERNAL, TrustLevel.INTERNAL}:
        raise HTTPException(status_code=403, detail="Verified caller trust is required.")


def _scope_from_verified_state(request: Request) -> tuple[str, str] | None:
    tenant_id = getattr(request.state, VERIFIED_TENANT_STATE_ATTR, "")
    org_id = getattr(request.state, VERIFIED_ORG_STATE_ATTR, "")
    if not tenant_id or not org_id:
        return None
    return str(tenant_id), str(org_id)


def _scope_from_stg_default() -> tuple[str, str] | None:
    tenant_id = os.environ.get(STG_DEFAULT_TENANT_ENV, "").strip()
    org_id = os.environ.get(STG_DEFAULT_ORG_ENV, "").strip()
    if not tenant_id or not org_id:
        return None
    return tenant_id, org_id


def _require_scope(request: Request, *, token_authenticated: bool) -> tuple[str, str]:
    """Return verified scope, allowing STG defaults only after token auth."""
    verify_scope = getattr(request.app.state, "verify_scope", _scope_from_verified_state)
    scope = verify_scope(request)
    if not scope and token_authenticated:
        scope = _scope_from_stg_default()
    if not scope:
        raise HTTPException(status_code=403, detail="Authenticated tenant and org scope is required.")
    tenant_id, org_id = scope
    if not tenant_id or not org_id:
        raise HTTPException(status_code=403, detail="Authenticated tenant and org scope is required.")
    return str(tenant_id), str(org_id)
