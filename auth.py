"""
auth.py — JWT authentication and RBAC permission checking for MCP Server.

Validates agent tokens issued by Auth-API (RS256 JWTs).
Checks permissions before tool execution based on the agent's role.

Flow:
  1. Agent sends request with Authorization: Bearer <jwt>
  2. Middleware validates JWT signature against Auth-API public key
  3. Extracts agent identity (name, role, department, permissions)
  4. Before tool execution, checks if agent has mcp:<server>:<tool> permission
  5. Propagates agent context to backend API calls via X-Agent-* headers

Configuration:
  AUTH_API_JWKS_URL   — JWKS endpoint (default: https://auth-api.arkturian.com/api/v1/auth/.well-known/jwks.json)
  AUTH_REQUIRE_JWT    — If "true", reject requests without JWT (default: "false" for transition)
  MCP_ERLAUBTE_AUD    — OAuth-Zielgruppen (client_id, fnmatch, kommagetrennt), die das
                        Gateway annimmt (#2976). Jedes andere aud gibt 401, auch bei leerer
                        Liste: dann gelten nur Tokens ohne aud (Agenten, Dienste, Login).
  MCP_SUPERADMIN_OHNE_SCOPE — "aus": ein Superadmin bekommt "*" nur mit eigenem Login,
                        nicht mit einem Token aus MCP_ERLAUBTE_AUD (Vorgabe: an).
  MCP_ABGELEHNTE_AUD  — aud-Muster, die nie gelten (Vorgabe: opencloud*); ai-realtime:*
                        gilt ausserdem immer als abgelehnt.
  JWKS_TTL_SECONDS    — Hoechstalter der JWKS im Speicher (Vorgabe 300).
"""

from __future__ import annotations

import contextvars
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import json as _json

import httpx
import jwt
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# ContextVars holding the current caller's identity for the duration of a
# single MCP request. FastMCP tool functions don't get the Request object
# directly, but ContextVars propagate through asyncio.copy_context() across
# every task spawned during the request, so tools can read these to forward
# the caller's JWT to upstream APIs (content-api, auth-api, etc.).
#
# JWTAuthMiddleware sets them; tools read via current_caller_jwt() etc.
_current_jwt: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_caller_jwt", default=""
)
_current_agent_name: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_caller_agent", default=""
)
# Nur fuer Tokens mit type=agent gesetzt. Ein Benutzer-Token traegt im
# Namensfeld seinen sub, das ist kein Agent (Storage-Freigabe-Scope, #5250).
_current_is_agent: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "current_caller_is_agent", default=False
)


# #3004: Klasse des gepruefften Tokens (user, oauth_access, agent, service, ...), damit
# der on-behalf-of-Tausch nur Menschen-Tokens tauscht.
_current_token_typ: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_caller_token_typ", default=""
)


def current_caller_ist_mensch() -> bool:
    """Ist das gepruefte Aufrufer-Token das eines Menschen (kein Agent, kein Dienst)?"""
    return bool(_current_jwt.get()) and _current_token_typ.get() not in ("agent", "service")


def current_caller_jwt() -> str:
    """Return the JWT (raw token string) of the agent that made the
    current MCP request, or empty string if anonymous/no auth.
    """
    return _current_jwt.get()


def current_caller_agent_name() -> str:
    """Return the agent_name claim of the current caller, or empty string."""
    return _current_agent_name.get()


def current_caller_verified_agent() -> str:
    """Name des Agenten, wenn das geprüfte JWT ein Agenten-Token ist, sonst ""."""
    return _current_agent_name.get() if _current_is_agent.get() else ""

logger = logging.getLogger("mcp-auth")

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

AUTH_API_JWKS_URL = os.getenv(
    "AUTH_API_JWKS_URL",
    "https://auth-api.arkturian.com/api/v1/auth/.well-known/jwks.json",
)
AUTH_API_ISSUER = os.getenv("AUTH_API_ISSUER", "auth-api.arkturian.com")
# Fail-closed (Steward/David 2026-08-28, dort #24): mit dem alten Default
# "false" fuehrte JEDER anonyme Aufrufer, der das Gateway erreicht, Werkzeuge
# aus -- verifiziert an einem mutierenden Aufruf, der ein Attribut wirklich
# schrieb, weil der Upstream-Helfer danach auf den Service-Key zurueckfaellt.
# Ein Uebergangs-Default, der Schreibzugriff ohne Identitaet erlaubt, ist die
# falsche Richtung; wer den alten Zustand braucht, setzt ihn ausdruecklich.
AUTH_REQUIRE_JWT = os.getenv("AUTH_REQUIRE_JWT", "true").lower() == "true"

# Paths that don't require auth. /docs und /openapi.json sind seit #2893 aus
# (MCP_DOCS=an schaltet sie ein, dann auch hier offen).
PUBLIC_PATHS = {"/", "/health", "/.well-known/mcp.json"}
if os.getenv("MCP_DOCS", "aus").strip().lower() in ("an", "1", "true", "yes"):
    PUBLIC_PATHS |= {"/docs", "/redoc", "/openapi.json"}

# JSON-RPC-Methoden, die OHNE Anmeldung durchgehen (#2893): nur der Handshake.
# Alles andere -- tools/list, resources/*, prompts/*, andere HTTP-Verben,
# Rumpf ohne JSON -- braucht ein gueltiges JWT mit Recht am Server.
OHNE_ANMELDUNG = {"initialize", "notifications/initialized", "ping"}


# ──────────────────────────────────────────────────────────────
# Agent context (stored in request.state.agent)
# ──────────────────────────────────────────────────────────────

@dataclass
class AgentContext:
    """Identity and permissions of the authenticated agent."""
    name: str = ""
    agent_type: str = ""
    role: str = ""
    department: str = ""
    tenant: str = ""
    # Fail-closed (#2890): ohne Recht kein Recht. Vorher ["*"] -- ein Kontext,
    # dem niemand Rechte zugewiesen hatte, durfte alles.
    permissions: list[str] = field(default_factory=list)
    authenticated: bool = False

    def has_permission(self, required: str) -> bool:
        """Check if this agent has a specific permission."""
        return any(_match_permission(p, required) for p in self.permissions)


def _match_permission(held: str, required: str) -> bool:
    """Wildcard permission matching."""
    if held == "*":
        return True
    held_parts = held.split(":")
    required_parts = required.split(":")
    for i, hp in enumerate(held_parts):
        if hp == "*":
            return True
        if i >= len(required_parts):
            return False
        if hp != required_parts[i]:
            return False
    return len(held_parts) == len(required_parts)


# ──────────────────────────────────────────────────────────────
# JWKS public key cache
# ──────────────────────────────────────────────────────────────

_public_key = None
_public_key_fetched_at = 0.0
# Hoechstens 5 min alt (#2972: Notfall-Rotation darf nicht eine Stunde haengen).
_PUBLIC_KEY_TTL = int(os.getenv("JWKS_TTL_SECONDS", "300") or 300)
# Alle Schluessel der JWKS je `kid` (#2972): vorher galt nur keys[0], ein Token mit
# dem zweiten Schluessel einer Rotation fiel durch, ein alter Schluessel blieb gueltig,
# solange er vorne stand.
_jwks_nach_kid: dict = {}
_JWKS_NACHLADEN_AB_S = 30.0          # unbekannter kid: hoechstens so oft neu holen


def _rsa_schluessel(key_data: dict):
    import base64
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
    from cryptography.hazmat.backends import default_backend
    n = int.from_bytes(base64.urlsafe_b64decode(key_data["n"] + "=="), "big")
    e = int.from_bytes(base64.urlsafe_b64decode(key_data["e"] + "=="), "big")
    return RSAPublicNumbers(e, n).public_key(default_backend())


async def _get_public_key():
    """JWKS holen und je `kid` ablegen; liefert den ersten Schluessel (fuer die
    Verfuegbarkeitspruefung im Middleware-Pfad). Gewaehlt wird je Token in
    _schluessel_fuer() nach seinem `kid`."""
    global _public_key, _public_key_fetched_at, _jwks_nach_kid

    if _public_key and (time.time() - _public_key_fetched_at < _PUBLIC_KEY_TTL):
        return _public_key

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(AUTH_API_JWKS_URL)
            res.raise_for_status()
            jwks = res.json()

        neu = {}
        erster = None
        for key_data in jwks.get("keys") or []:
            if key_data.get("kty") not in (None, "RSA"):
                continue
            k = _rsa_schluessel(key_data)
            neu[str(key_data.get("kid") or "")] = k
            erster = erster or k
        if erster is None:
            raise ValueError("JWKS ohne RSA-Schluessel")
        _jwks_nach_kid = neu
        _public_key = erster
        _public_key_fetched_at = time.time()
        logger.info("JWKS fetched from %s: %d key(s) %s", AUTH_API_JWKS_URL, len(neu), sorted(neu))
        return _public_key
    except Exception as exc:
        logger.error("Failed to fetch JWKS from %s: %s", AUTH_API_JWKS_URL, exc)
        return _public_key  # return cached key if available


def _kid_aus(token: str) -> Optional[str]:
    try:
        kid = jwt.get_unverified_header(token).get("kid")
    except Exception:
        return None
    return str(kid) if kid else None


async def _jwks_fuer_kid(kid: Optional[str]) -> None:
    """Kennt der Zwischenspeicher den `kid` des Tokens nicht, einmal neu holen
    (Rotation), aber nicht oefter als alle 30 s (kein Hebel fuer Last auf auth-api)."""
    global _public_key_fetched_at
    if not kid or not _jwks_nach_kid or kid in _jwks_nach_kid:
        return
    if time.time() - _public_key_fetched_at < _JWKS_NACHLADEN_AB_S:
        return
    _public_key_fetched_at = 0.0
    await _get_public_key()


def _schluessel_fuer(token: str, public_key):
    """Der Schluessel, mit dem dieses Token geprueft wird. Mit `kid`: genau der aus
    der JWKS, ein unbekannter `kid` wird abgewiesen. Ohne `kid`: nur, wenn die JWKS
    genau einen Schluessel hat. Ist (noch) keine JWKS geladen, gilt `public_key`."""
    if not _jwks_nach_kid:
        return public_key
    kid = _kid_aus(token)
    if kid:
        if kid not in _jwks_nach_kid:
            raise jwt.InvalidTokenError(f"unknown signing key (kid {kid!r})")
        return _jwks_nach_kid[kid]
    if len(_jwks_nach_kid) == 1:
        return next(iter(_jwks_nach_kid.values()))
    raise jwt.InvalidTokenError("token without kid, JWKS has several keys")


# Zielgruppen (aud), die am Gateway nichts zu suchen haben (#2890): Tokens, die
# auth-api fuer ANDERE Anwendungen ausstellt (OpenCloud-Web u. a.). Muster mit
# fnmatch, kommagetrennt in MCP_ABGELEHNTE_AUD.
_ABGELEHNTE_AUD = [m.strip() for m in os.getenv("MCP_ABGELEHNTE_AUD", "opencloud*").split(",") if m.strip()]
# Voice-Grants (create_realtime_grant_token: aud "ai-realtime:<profil>", ohne type)
# gelten am Gateway nie, auch nicht ueber MCP_ERLAUBTE_AUD (#2976, Befund 76).
_IMMER_ABGELEHNTE_AUD = ("ai-realtime:*",)
# Token-Arten am Gateway (#2976): ohne type = Login- bzw. OAuth-Token des Menschen
# (heute), user/oauth_access = dieselben, sobald auth-api type setzt (#2970).
# Abgewiesen: ephemeral_capability, realtime_grant und jede unbekannte Art.
_ZUGELASSENE_TYPEN = frozenset({"", "agent", "service", "user", "oauth_access"})
# Positive Liste der OAuth-Zielgruppen (client_id), fnmatch, kommagetrennt, je Gateway
# (#2976, Entscheid Cloud 15.11.2026). Ein Token mit aud aus der Liste gilt wie ein
# eigenes Login (Superadmin "*", sonst seine ausdruecklichen mcp:-Scopes). Jedes andere
# aud gibt 401 -- auch bei LEERER Liste: dann gelten nur Tokens ohne aud (Agenten,
# Dienste, eigenes Login). Das schliesst die Kette ueber anonym per DCR registrierte
# Clients (Befund 76): ein Superadmin, der sich dort anmeldet, gab dem Client "*".
_ERLAUBTE_AUD = [m.strip() for m in os.getenv("MCP_ERLAUBTE_AUD", "").split(",") if m.strip()]
# Superadmin ohne mcp:-Scope bekommt mit einem Token aus MCP_ERLAUBTE_AUD "*" (die
# ChatGPT-Connectoren des Betreibers tragen keine Scopes). Abschalten:
# MCP_SUPERADMIN_OHNE_SCOPE=aus, dann gilt "*" nur fuer das eigene Login (ohne aud).
_SUPERADMIN_OHNE_SCOPE = os.getenv("MCP_SUPERADMIN_OHNE_SCOPE", "an").strip().lower() not in ("aus", "0", "false", "no")


def _aud_werte(claims: dict) -> list:
    aud = claims.get("aud")
    return [str(a) for a in (aud if isinstance(aud, list) else ([aud] if aud else []))]


def _passt(werte: list, muster) -> bool:
    import fnmatch
    return any(fnmatch.fnmatchcase(a.lower(), m.lower()) for a in werte for m in muster)


def _token_abweisen(claims: dict) -> Optional[str]:
    """None = Token darf weiter, sonst der Grund fuer 401 (#2976).

    Reihenfolge: Token-Art, Voice-Grant, gesperrte aud (#2890), positive aud-Liste,
    min_agent_iat fuer Agenten- und Dienst-Tokens (#2972)."""
    typ = str(claims.get("type") or "")
    if typ not in _ZUGELASSENE_TYPEN:
        return f"token type {typ!r} not accepted"
    auds = _aud_werte(claims)
    if _passt(auds, _IMMER_ABGELEHNTE_AUD):
        return "voice grant not accepted"
    if _aud_abgelehnt(claims):
        return "audience not accepted"
    if auds and not _passt(auds, _ERLAUBTE_AUD):
        return "audience not allowed (MCP_ERLAUBTE_AUD)"
    if typ in ("agent", "service"):
        grund = _zu_alt(claims)
        if grund:
            return grund
    return None


def _aud_abgelehnt(claims: dict) -> bool:
    import fnmatch
    aud = claims.get("aud")
    werte = aud if isinstance(aud, list) else ([aud] if aud else [])
    return any(fnmatch.fnmatchcase(str(a).lower(), m.lower()) for a in werte for m in _ABGELEHNTE_AUD)


def _rechte_aus_claims(claims: dict) -> list:
    """Die Rechte eines gepruften Tokens -- nie aus dem FEHLEN eines Claims (#2890).

    Vorher `claims.get("permissions", ["*"])`: auth-api stellt Benutzer-, OIDC-
    und OAuth-Tokens ohne `permissions` aus, jedes gueltige Benutzer-Token hatte
    damit am Gateway alle Rechte.
      - Agent und Dienst (type agent/service): genau ihr `permissions`-Claim.
      - Mensch mit OAuth-Token: je Scope `mcp:<server>` das Recht mcp:<server>:*.
      - Superadmin (Betreiber) mit eigenem Login-Token (ohne aud): "*"; mit
        OAuth-Token ohne mcp:-Scope nur, wenn dessen aud in MCP_ERLAUBTE_AUD steht
        (#2976). Jedes andere aud weist _token_abweisen ohnehin ab.
      - sonst: keine Rechte.
    """
    typ = claims.get("type")
    if typ in ("agent", "service"):
        p = claims.get("permissions")
        return [str(x) for x in p] if isinstance(p, list) else []
    scopes = str(claims.get("scope") or "").split()
    aus_scopes = sorted({f"mcp:{sc[4:]}:*" for sc in scopes
                         if sc.startswith("mcp:") and sc.count(":") == 1 and len(sc) > 4})
    if aus_scopes:
        return aus_scopes
    if claims.get("is_superadmin") is True:
        auds = _aud_werte(claims)
        if not auds or (_SUPERADMIN_OHNE_SCOPE and _passt(auds, _ERLAUBTE_AUD)):
            return ["*"]
    return []


def _decode_jwt(token: str, public_key) -> dict:
    """Decode and validate a JWT token.

    `verify_aud` is off deliberately. PyJWT raises InvalidAudienceError when a
    token *carries* an `aud` claim and the verifier passes no expected audience
    — silence is not an option, it is an error. auth-api stamps `aud=client_id`
    on OAuth access tokens (oauth_service.py), and that client_id is minted per
    client by dynamic registration, so a resource server cannot know the set of
    valid values in advance.

    The practical effect before this change: agents using a static JWT (no
    `aud`) worked, while every OAuth client was rejected with
    "401 Invalid token: Invalid audience" — ChatGPT could not use any MCP
    server at all, Claude Code could (Alex 2026-08-04).

    Dropping the check does not widen trust: the token is still proven to come
    from auth-api by RS256 signature over the JWKS key plus the issuer check,
    expiry is still enforced, and authorisation is decided further down by the
    per-agent RBAC lookup on `sub`. `aud` only records which client the token
    was issued to — it was never the gate here.

    The cleaner long-term shape is RFC 8707: have auth-api stamp the resource
    (this server) as the audience and verify it explicitly. That touches every
    auth-api consumer, so it belongs in its own change, not in an outage fix.
    """
    claims = jwt.decode(
        token,
        _schluessel_fuer(token, public_key),
        algorithms=["RS256"],
        issuer=AUTH_API_ISSUER,
        options={"verify_exp": True, "verify_aud": False},
    )
    _reject_if_revoked(claims)
    return claims


# Interims-Widerruf (27.08.2026): auth-api prueft Agent-Tokens rein per JWKS und
# hat noch keine Deny-Liste (AuthAPI baut revoked_agent_jtis + Poll-Endpunkt).
# Bis dahin: feste jti-Liste + Env-Override REVOKED_AGENT_JTIS (kommagetrennt),
# damit der naechste Fall keinen Release braucht. Gleiche Liste wie in cloud-api.
_REVOKED_JTIS: set = set()   # Notnagel ohne auth-api; Regelweg ist revoked_poll_loop (Deny-Liste, seit 27.08.)


# Poll-Seite der auth-api Deny-Liste (Spur B, AuthAPI 27.08.): oeffentlicher
# Endpunkt neben JWKS, Zeilen werden nie erneut ausgegeben, ``since=<as_of>``
# ist die Fortschrittsmarke. Menge im Prozess + Spiegel auf Platte; ein
# Poll-Ausfall leert nichts.
_REVOKED_POLL_SECONDS = int(os.getenv("REVOKED_JTIS_POLL_SECONDS", "30") or 30)
_REVOKED_STATE = os.getenv("REVOKED_JTIS_STATE", "/var/lib/mcp-server/revoked_jtis.json")
_revoked_polled: set = set()
_revoked_as_of = ""
_revoked_loaded = False
# min_agent_iat der auth-api (#2972): Agenten- und Dienst-Tokens, die davor
# ausgestellt wurden, gelten nicht mehr. Spiegelt den AKTUELLEN Wert, auch ein
# Absenken oder null (wie content-api/issue-api); fehlt der Schluessel in der
# Antwort (alter Stand der auth-api), bleibt der letzte Stand.
_min_agent_iat: Optional[int] = None


def _als_epoch(wert) -> Optional[int]:
    if wert is None or wert == "" or isinstance(wert, bool):
        return None
    if isinstance(wert, (int, float)):
        return int(wert)
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(wert).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except (TypeError, ValueError):
        logger.warning("revoked_jtis: min_agent_iat unlesbar (%r), ignoriert", wert)
        return None


def _zu_alt(claims: dict) -> Optional[str]:
    _revoked_load()
    if _min_agent_iat is None:
        return None
    iat = claims.get("iat")
    if isinstance(iat, (int, float)) and not isinstance(iat, bool) and iat < _min_agent_iat:
        return "token too old (iat before min_agent_iat), mint a new one"
    return None


def _revoked_endpoint() -> str:
    return AUTH_API_JWKS_URL.rsplit("/", 1)[0] + "/revoked-agent-jtis"


def _revoked_load() -> None:
    global _revoked_polled, _revoked_as_of, _revoked_loaded, _min_agent_iat
    if _revoked_loaded:
        return
    _revoked_loaded = True
    try:
        import json
        with open(_REVOKED_STATE) as f:
            d = json.load(f)
        _revoked_polled = set(d.get("revoked") or [])
        _revoked_as_of = str(d.get("as_of") or "")
        _min_agent_iat = _als_epoch(d.get("min_agent_iat"))
    except Exception:
        pass


def _revoked_save() -> None:
    try:
        import json
        os.makedirs(os.path.dirname(_REVOKED_STATE), exist_ok=True)
        tmp = _REVOKED_STATE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"revoked": sorted(_revoked_polled), "as_of": _revoked_as_of,
                       "min_agent_iat": _min_agent_iat}, f)
        os.replace(tmp, _REVOKED_STATE)
    except Exception as e:
        logger.warning("revoked_jtis: Spiegel nicht geschrieben: %s", e)


async def revoked_poll_once(client=None) -> int:
    """Einmal abholen; liefert Anzahl NEUER jtis. Wirft nicht."""
    global _revoked_as_of, _min_agent_iat
    _revoked_load()
    try:
        import httpx
        params = {"since": _revoked_as_of} if _revoked_as_of else {}
        c = client or httpx.AsyncClient(timeout=6)
        try:
            r = await c.get(_revoked_endpoint(), params=params)
        finally:
            if client is None:
                await c.aclose()
        if r.status_code != 200:
            logger.warning("revoked_jtis: Poll -> HTTP %s", r.status_code)
            return 0
        d = r.json() or {}
        neu = 0
        for e in d.get("revoked_jtis") or []:
            j = str((e or {}).get("jti") or "")
            if j and j not in _revoked_polled:
                _revoked_polled.add(j)
                neu += 1
        _revoked_as_of = str(d.get("as_of") or _revoked_as_of)
        geaendert = False
        if "min_agent_iat" in d:             # fehlt der Schluessel: letzter Stand bleibt
            geliefert = _als_epoch(d.get("min_agent_iat"))
            if geliefert != _min_agent_iat:
                if geliefert is not None and geliefert > time.time() + 300:
                    logger.error("revoked_jtis: min_agent_iat liegt in der Zukunft, auch frisch "
                                 "gepraegte Agenten-Tokens werden abgewiesen")
                _min_agent_iat = geliefert
                geaendert = True
        if neu or geaendert or not os.path.exists(_REVOKED_STATE):
            _revoked_save()
        if neu:
            logger.warning("revoked_jtis: %d neue widerrufene Agent-Token (gesamt %d)", neu, len(_revoked_polled))
        return neu
    except Exception as e:
        logger.warning("revoked_jtis: Poll fehlgeschlagen: %s", e)
        return 0


async def revoked_poll_loop() -> None:
    import asyncio
    await asyncio.sleep(5)
    while True:
        await revoked_poll_once()
        await asyncio.sleep(_REVOKED_POLL_SECONDS)


def _reject_if_revoked(claims: dict) -> None:
    jti = str(claims.get("jti") or "")
    if not jti:
        return
    _revoked_load()
    extra = {x.strip() for x in os.getenv("REVOKED_AGENT_JTIS", "").split(",") if x.strip()}
    if jti in _REVOKED_JTIS or jti in extra or jti in _revoked_polled:
        logger.warning("JWT abgewiesen: jti %s widerrufen (sub=%s)", jti, claims.get("sub"))
        raise jwt.InvalidTokenError("token revoked")


# ──────────────────────────────────────────────────────────────
# MCP path → permission mapping
# ──────────────────────────────────────────────────────────────

# Server, an denen Platzhalter ("*", "mcp:*", "mcp:*:*", "mcp:*:<tool>") nicht
# zaehlen: durch kommt nur, wer mcp:<server>:* oder mcp:<server>:<tool>
# ausdruecklich traegt. server.py meldet hier die Durchreichen an (Entscheid
# Alex 26.10.2026: p3-buchhaltung nur mit ausdruecklichem Recht).
EXPLICIT_ONLY_SERVERS: set[str] = set()
# Server, deren Upstream-Helfer einen BETREIBER-Hauptschluessel statt des
# Aufrufer-JWT schicken (#2890: business, oneal-storage, codepilot, artrack,
# story). Ein Platzhalterrecht reicht dort nicht -- sonst ruft jeder mit "*"
# die Hauptschluessel-Dienste ueber Mandantengrenzen hinweg auf. Je Gateway
# per MCP_EXPLIZIT_SERVER aenderbar (leer = keiner).
_HAUPTSCHLUESSEL_SERVER = os.getenv("MCP_EXPLIZIT_SERVER", "business,oneal-storage,codepilot,artrack,story")
EXPLICIT_ONLY_SERVERS.update(s.strip().lower() for s in _HAUPTSCHLUESSEL_SERVER.split(",") if s.strip())


# Mandanten-Schranke (Cloud-Security AP 2.2, 15.11.2026): An Servern, deren Upstream
# einen BETREIBER-Hauptschluessel bekommt (business, storage, oneal ...), zaehlt neben
# dem Recht auch der Mandant aus dem JWT. Sonst erreichen Agenten fremder Mandanten mit
# einer globalen Rolle (engineering/developer traegt mcp:business:* usw.) die
# Buchhaltung und den Speicher des Betreibers. Form: "srv1,srv2:mand1,mand2;srv3:mand3".
# Vorgabe LEER: dieselbe Codebasis laeuft als Gateway jeder Kundeninstanz mit eigenem
# Mandanten -- gesetzt wird die Schranke je Gateway per MCP_MANDANTEN_SCHRANKE.
# Ohne tenant-Claim (Alt-Tokens eigener Agenten, Connectoren) greift sie nicht.
def _schranke_laden(spec: str) -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    for teil in (spec or "").split(";"):
        if ":" not in teil:
            continue
        server, mandanten = teil.split(":", 1)
        erlaubt = frozenset(m.strip().lower() for m in mandanten.split(",") if m.strip())
        for srv in server.split(","):
            if srv.strip() and erlaubt:
                out[srv.strip().lower()] = erlaubt
    return out


MANDANTEN_SCHRANKE: dict[str, frozenset[str]] = _schranke_laden(os.getenv("MCP_MANDANTEN_SCHRANKE", ""))


def mandant_gesperrt(mcp_server: str, tenant: str, *, ist_agent: bool = True,
                     user_tids=(), superadmin: bool = False) -> bool:
    """True, wenn ein Aufrufer mit diesem Mandanten den Server nicht erreichen darf.

    Agenten-/Dienst-Tokens: der einzelne `tenant`-Claim; ohne Claim (Alt-Tokens eigener
    Agenten) unveraendert durchgelassen. Menschen-/OAuth-Tokens tragen keinen `tenant`,
    sondern `roles[].tid` (Cloud-Security AP 2.2): durch nur Superadmin oder wer einen
    zugelassenen Mandanten in seinen Rollen hat -- sonst fail-closed."""
    erlaubt = MANDANTEN_SCHRANKE.get((mcp_server or "").lower())
    if not erlaubt:
        return False
    t = (tenant or "").strip().lower()
    if t:
        return t not in erlaubt
    if ist_agent or superadmin:
        return False
    tids = {str(x or "").strip().lower() for x in (user_tids or ())}
    return not (tids & erlaubt)


def _rollen_mandanten(claims: dict) -> list:
    return [r.get("tid") for r in (claims.get("roles") or []) if isinstance(r, dict) and r.get("tid")]


# Server, die JEDER angemeldete Agent dieser Instanz nutzen darf, ohne eigenes
# mcp:<server>:*-Recht (Alex 08.11.2026 zu llmtime, #2679: "die Agents sollen das
# Recht haben" -- und es soll einfach funktionieren, ohne Rollenpflege je neuem
# Dienst). Je Gateway per MCP_OFFENE_SERVER (kommagetrennt), Vorgabe leer: eine
# Kundeninstanz oeffnet so nichts, was sie nicht selbst einstellt. Die
# Anmeldung (gueltiges JWT) bleibt Pflicht; EXPLICIT_ONLY_SERVERS gewinnt.
OFFENE_SERVER: set[str] = {
    s.strip().lower() for s in os.getenv("MCP_OFFENE_SERVER", "").split(",") if s.strip()
}


# Server mit Rechte-STUFEN statt Rechten je Werkzeug (#2441, person-api): Werkzeug ->
# Stufe, z. B. {"person": {"persons_list": "read", "person_create": "write"}}. Ein
# Werkzeug ist dann mit mcp:<server>:<stufe> ODER mcp:<server>:<werkzeug> frei.
# Die Stufen gelten woertlich, ohne Vererbung -- wie in der person-api selbst.
# server.py meldet die Zuordnung beim Einhaengen an.
WERKZEUG_STUFEN: dict[str, dict[str, str]] = {}


def _werkzeug_erlaubt(perms: list[str], mcp_server: str, tool_name: str) -> bool:
    """Darf ein Agent mit `perms` (schon nach _permissions_for_server) das Werkzeug rufen?"""
    if any(_match_permission(p, f"mcp:{mcp_server}:{tool_name}") for p in perms):
        return True
    stufe = WERKZEUG_STUFEN.get(mcp_server, {}).get(tool_name)
    return stufe is not None and any(_match_permission(p, f"mcp:{mcp_server}:{stufe}") for p in perms)


def _jsonrpc_teile(body_json) -> list:
    """Einzelne JSON-RPC-Nachrichten; ein Stapel (Liste) wird aufgefaechert."""
    teile = body_json if isinstance(body_json, list) else [body_json]
    return [t for t in teile if isinstance(t, dict)]


def _aufgerufene_werkzeuge(body_json) -> list:
    """Namen aller tools/call-Aufrufe im Rumpf, auch aus einem Stapel. Ein Aufruf
    ohne Namen ergibt "" -- der Aufrufer lehnt ihn ab."""
    return [str((t.get("params") if isinstance(t.get("params"), dict) else {}).get("name") or "")
            for t in _jsonrpc_teile(body_json) if t.get("method") == "tools/call"]


# Werkzeug-Parameter, die entfernt sind und die FastMCP sonst STILL verwerfen
# wuerde (#2894; FastMCP ignoriert unbekannte Argumente). business_*(api_key=...)
# liefe danach mit dem Schluessel des Gateways -- also womoeglich im falschen
# Mandanten, denn business-api bindet den Mandanten an den Schluessel. Lieber eine
# klare Ablehnung als ein Aufruf in einem anderen Mandanten.
ENTFERNTE_ARGUMENTE = {
    "business": {
        "api_key": ("api_key ist kein Werkzeug-Parameter mehr (#2894): die business-Werkzeuge "
                    "laufen mit dem Mandanten dieses Gateways. Fuer einen anderen Mandanten "
                    "bitte Cloud fragen, den Schluessel nicht im Aufruf mitgeben."),
    },
}


def _entfernte_argumente(mcp_server: str, body_json) -> str:
    """Grund, wenn ein tools/call ein entferntes Argument mitschickt, sonst ""."""
    regeln = ENTFERNTE_ARGUMENTE.get(mcp_server) or {}
    if not regeln:
        return ""
    for t in _jsonrpc_teile(body_json):
        if t.get("method") != "tools/call":
            continue
        params = t.get("params") if isinstance(t.get("params"), dict) else {}
        args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        for name, grund in regeln.items():
            if name in args:
                return grund
    return ""


def _offen(mcp_server: str) -> bool:
    return mcp_server in OFFENE_SERVER and mcp_server not in EXPLICIT_ONLY_SERVERS


def _permissions_for_server(permissions: list[str], mcp_server: str) -> list[str]:
    """Die Rechte, die an diesem Server zaehlen."""
    if mcp_server in EXPLICIT_ONLY_SERVERS:
        prefix = f"mcp:{mcp_server}:"
        return [p for p in permissions if p.startswith(prefix)]
    return permissions


def gegriffenes_recht(permissions: list[str], mcp_server: str, tool_name: str) -> str:
    """Das GENAUESTE Recht, das diesen Aufruf erlaubt (#2806): mcp:<s>:<tool> bzw.
    die Stufe vor mcp:<s>:* vor mcp:*:* / mcp:* vor *. Steht hier ein Platzhalter,
    war er fuer den Aufruf noetig. "offen" fuer offene Server; dieselbe Pruefung
    wie im Tor (_permissions_for_server, _match_permission, WERKZEUG_STUFEN)."""
    import aufruf_protokoll as _ap
    if mcp_server and _offen(mcp_server):
        return "offen"
    perms = _permissions_for_server(permissions, mcp_server)
    ziele = [f"mcp:{mcp_server}:{tool_name}"]
    stufe = WERKZEUG_STUFEN.get(mcp_server, {}).get(tool_name)
    if stufe:
        ziele.append(f"mcp:{mcp_server}:{stufe}")
    treffer = [p for p in perms if any(_match_permission(p, z) for z in ziele)]
    return max(treffer, key=_ap.genauigkeit) if treffer else "?"


def _aufrufe_vermerken(agent: "AgentContext", mcp_server: str, body_bytes) -> None:
    """Erlaubte Aufrufe zaehlen (aufruf_protokoll). Nie Argumente; ein Fehler
    hier haelt keinen Aufruf auf."""
    try:
        import aufruf_protokoll as _ap
        werkzeuge = _aufgerufene_werkzeuge(_json.loads(body_bytes)) if body_bytes else []
        for werkzeug in werkzeuge or [""]:
            _ap.protokoll().vermerken(agent.name, agent.role, mcp_server, werkzeug,
                                      gegriffenes_recht(agent.permissions, mcp_server, werkzeug))
    except Exception:
        logger.debug("aufruf-protokoll: vermerken gescheitert", exc_info=True)


def _extract_mcp_server_from_path(path: str) -> str:
    """Extract MCP server name from request path.

    /storage/...  → storage
    /business/... → business
    /cloud/...    → cloud
    """
    parts = path.strip("/").split("/")
    return parts[0] if parts else ""


# ──────────────────────────────────────────────────────────────
# FastAPI Middleware
# ──────────────────────────────────────────────────────────────

class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Validate JWT tokens and inject AgentContext into request.state."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        pfad = request.url.path
        if pfad in PUBLIC_PATHS or (request.method == "GET" and pfad.startswith("/.well-known/")) \
                or request.method == "OPTIONS":
            request.state.agent = AgentContext()
            return await call_next(request)

        # Ohne Anmeldung geht nur der Handshake (#2893, fail-closed). Vorher
        # galt die Pruefung nur fuer tools/call: GET, andere Verben, Rumpf ohne
        # JSON und tools/list, resources/read, prompts/get liefen ohne JWT --
        # auch an EXPLICIT_ONLY_SERVERS und durch die Durchreiche. Fehlt das
        # Token, antwortet das Gateway 403 (nicht 401: kein OAuth-Ausloeser in
        # Claude Code), ein GET ohne Token 405 wie ein zustandsloser MCP-Server.
        body_bytes = None
        jsonrpc_method = ""
        methoden: list = []
        if request.method == "POST":
            try:
                body_bytes = await request.body()
                request._body = body_bytes  # cache for downstream
                body_json = _json.loads(body_bytes)
                # Ein Stapel mit einem Werkzeugaufruf zaehlt als Werkzeugaufruf
                # (fail-closed). Vorher warf `.get` auf der Liste, und der Stapel
                # lief ohne JWT-Pruefung weiter; mcp 1.21/1.26 lehnt Stapel heute
                # selbst mit 400 ab, eine kuenftige Fassung soll hier kein Loch finden.
                methoden = [t.get("method", "") for t in _jsonrpc_teile(body_json)]
                jsonrpc_method = "tools/call" if "tools/call" in methoden else (
                    methoden[0] if len(methoden) == 1 else "")
            except Exception:
                methoden = []

        is_tool_call = jsonrpc_method == "tools/call"
        if is_tool_call and body_bytes:
            try:
                _koerper = _json.loads(body_bytes)
                _grund = _entfernte_argumente(_extract_mcp_server_from_path(request.url.path), _koerper)
            except Exception:
                _koerper, _grund = None, ""
            if _grund:
                return JSONResponse({"jsonrpc": "2.0",
                                     "id": _koerper.get("id") if isinstance(_koerper, dict) else None,
                                     "error": {"code": -32602, "message": _grund}}, status_code=400)
        if request.method == "POST" and methoden and all(m in OHNE_ANMELDUNG for m in methoden):
            request.state.agent = AgentContext()
            return await call_next(request)

        # Extract token from Authorization header
        auth_header = request.headers.get("authorization", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

        if not token:
            if AUTH_REQUIRE_JWT:
                if request.method == "GET":
                    return JSONResponse({"error": "Method Not Allowed"}, status_code=405)
                return JSONResponse(
                    {"error": "Authorization required"},
                    status_code=403,  # 403 not 401 — avoids OAuth trigger
                )
            # No token = anonymous (transition period). SICHTBAR machen, wer
            # das ueberhaupt nutzt: ohne diese Zeile ist "wir sehen keine
            # anonymen Aufrufe" kein Befund, sondern eine fehlende Messung
            # (#1462). Werkzeugname und Aufrufer-IP genuegen zur Entscheidung,
            # ob der strikte Modus jemanden abschneidet.
            try:
                _tool = ""
                if body_bytes:
                    _tool = ",".join(_aufgerufene_werkzeuge(_json.loads(body_bytes)))
                _peer = request.headers.get("x-forwarded-for") or (
                    request.client.host if request.client else "?")
                logger.warning(
                    "anonymous tools/call zugelassen (AUTH_REQUIRE_JWT=false): tool=%s von=%s",
                    _tool or "?", _peer)
            except Exception:
                pass
            request.state.agent = AgentContext()
            return await call_next(request)

        # Validate JWT
        try:
            public_key = await _get_public_key()
            if not public_key:
                # Can't validate — allow in transition mode, deny in strict mode
                if AUTH_REQUIRE_JWT:
                    return JSONResponse(
                        {"error": "Auth service unavailable (JWKS fetch failed)"},
                        status_code=503,
                    )
                request.state.agent = AgentContext()
                return await call_next(request)

            await _jwks_fuer_kid(_kid_aus(token))
            claims = _decode_jwt(token, public_key)
            _grund = _token_abweisen(claims)
            if _grund:
                logger.warning("JWT abgewiesen: %s (type=%r aud=%r sub=%s)", _grund,
                               claims.get("type"), claims.get("aud"), claims.get("sub"))
                return JSONResponse({"error": f"Invalid token: {_grund}"}, status_code=401)

            agent = AgentContext(
                name=claims.get("agent_name", claims.get("sub", "")),
                agent_type=claims.get("agent_type", ""),
                role=claims.get("role", ""),
                department=claims.get("department", ""),
                tenant=claims.get("tenant", ""),
                permissions=_rechte_aus_claims(claims),
                authenticated=True,
            )
            request.state.agent = agent

            # Stash the raw JWT into ContextVars so tool functions further
            # downstream can read it for upstream API forwarding. Reset on
            # response (no token-leak between requests).
            _jwt_token = _current_jwt.set(token)
            _name_token = _current_agent_name.set(agent.name)
            _is_agent_token = _current_is_agent.set(
                claims.get("type") == "agent" and bool(claims.get("agent_name"))
            )
            _typ_token = _current_token_typ.set(str(claims.get("type") or "user"))

            # Permission check: server-level + tool-level
            mcp_server = _extract_mcp_server_from_path(request.url.path)
            if mcp_server and mandant_gesperrt(
                mcp_server, agent.tenant,
                ist_agent=claims.get("type") in ("agent", "service"),
                user_tids=_rollen_mandanten(claims),
                superadmin=claims.get("is_superadmin") is True,
            ):
                logger.warning(
                    "Mandanten-Schranke: agent=%s tenant=%s mcp:%s abgewiesen",
                    agent.name, agent.tenant, mcp_server,
                )
                return JSONResponse(
                    {
                        "error": f"Permission denied: tenant '{agent.tenant}' has no access to mcp:{mcp_server}",
                        "agent": agent.name,
                        "code": "mandant_nicht_zugelassen",
                    },
                    status_code=403,
                )
            if mcp_server and _offen(mcp_server):
                pass  # offener Server: jedes gueltige Agenten-JWT genuegt
            elif mcp_server:
                # 1. Server-level: does agent have ANY permission on this MCP?
                # Recognized claims:
                #   "*"                       — global wildcard
                #   "mcp:*"                   — all MCPs (2-part)
                #   "mcp:*:*"                 — all MCPs (3-part, architect)
                #   "mcp:<server>:*"          — full access to this MCP
                #   "mcp:<server>:<tool>"     — any specific tool on this MCP
                # The literal 2-part vs 3-part mismatch (mcp:* vs mcp:*:*)
                # silently broke architect-role agents — both forms now valid.
                server_prefix = f"mcp:{mcp_server}:"
                perms = _permissions_for_server(agent.permissions, mcp_server)
                has_any = any(
                    p == "*" or p == "mcp:*" or p == "mcp:*:*" or p.startswith(server_prefix)
                    for p in perms
                )
                if not has_any:
                    logger.warning(
                        "RBAC denied (server): agent=%s role=%s mcp:%s",
                        agent.name, agent.role, mcp_server,
                    )
                    return JSONResponse(
                        {
                            "error": f"Permission denied: agent '{agent.name}' (role '{agent.role}') "
                                     f"does not have access to mcp:{mcp_server}",
                            "agent": agent.name,
                            "role": agent.role,
                            "required": f"mcp:{mcp_server}:*",
                        },
                        status_code=403,
                    )

                # 2. Tool-level: check specific tool from the body we already parsed
                if not any(_match_permission(p, f"mcp:{mcp_server}:*") for p in perms) and body_bytes:
                    try:
                        body_json = _json.loads(body_bytes)
                        # Jeder Aufruf eines Stapels einzeln.
                        for tool_name in _aufgerufene_werkzeuge(body_json):
                            required_perm = f"mcp:{mcp_server}:{tool_name}"
                            stufe = WERKZEUG_STUFEN.get(mcp_server, {}).get(tool_name)
                            if stufe:
                                required_perm += f" | mcp:{mcp_server}:{stufe}"
                            if not tool_name or not _werkzeug_erlaubt(perms, mcp_server, tool_name):
                                logger.warning(
                                    "RBAC denied (tool): agent=%s role=%s perm=%s",
                                    agent.name, agent.role, required_perm,
                                )
                                return JSONResponse(
                                    {
                                        "jsonrpc": "2.0",
                                        "id": body_json.get("id") if isinstance(body_json, dict) else None,
                                        "error": {
                                            "code": -32600,
                                            "message": f"Permission denied: agent '{agent.name}' "
                                                       f"(role '{agent.role}') cannot call {required_perm}",
                                        },
                                    },
                                    status_code=403,
                                )
                    except Exception:
                        # Fail-closed: wirft die Pruefung unerwartet, wird der Aufruf
                        # abgewiesen statt ungeprueft durchgelassen (vorher `pass`).
                        logger.exception(
                            "RBAC tool check failed, denied: agent=%s role=%s mcp:%s",
                            agent.name, agent.role, mcp_server,
                        )
                        return JSONResponse(
                            {
                                "jsonrpc": "2.0",
                                "id": None,
                                "error": {
                                    "code": -32600,
                                    "message": f"Permission check failed for agent '{agent.name}' "
                                               f"on mcp:{mcp_server}; call denied",
                                },
                            },
                            status_code=403,
                        )

            # Erlaubt: Aufrufer, Server, Werkzeug und gegriffenes Recht zaehlen (#2806).
            # Nur Werkzeugaufrufe -- tools/list & Co. waeren sonst leere Zeilen.
            if is_tool_call:
                _aufrufe_vermerken(agent, mcp_server, body_bytes)

        except jwt.ExpiredSignatureError:
            return JSONResponse({"error": "Token expired"}, status_code=401)
        except jwt.InvalidTokenError as e:
            return JSONResponse({"error": f"Invalid token: {e}"}, status_code=401)

        try:
            return await call_next(request)
        finally:
            # Reset ContextVars after request — prevents JWT bleed between calls
            try:
                _current_jwt.reset(_jwt_token)
                _current_agent_name.reset(_name_token)
                _current_is_agent.reset(_is_agent_token)
                _current_token_typ.reset(_typ_token)
            except (LookupError, NameError):
                pass


# ──────────────────────────────────────────────────────────────
# Context propagation headers
# ──────────────────────────────────────────────────────────────

def get_agent_headers(request: Request) -> dict[str, str]:
    """Extract agent context from request and return headers for backend API calls.

    Add these headers to _fetch_json calls so backend APIs know who's calling.
    """
    agent: AgentContext = getattr(request.state, "agent", None) or AgentContext()
    if not agent.authenticated:
        return {}
    from agent_identity_header import set_agent_name
    return set_agent_name({
        "X-Agent-Role": agent.role,
        "X-Agent-Department": agent.department,
        "X-Agent-Tenant": agent.tenant or "",
    }, agent.name)
