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
  AUTH_API_ISSUER     — iss der auth-api (Vorgabe auth-api.arkturian.com, ohne https://)
  AUTH_API_JWKS_URL   — JWKS endpoint (default: https://auth-api.arkturian.com/api/v1/auth/.well-known/jwks.json);
                        der Widerruf (revoked-agent-jtis) liegt daneben. Gilt, solange
                        MCP_JWT_BASE_URL leer ist.
  MCP_JWT_BASE_URL    — Abrufadresse der auth-api fuer JWKS und Widerruf (agentos_jwt, #3161);
                        auf arkturian http://localhost:8050.
  MCP_JWT_STATE_DIR   — Widerrufszustand des Pruefmoduls (Vorgabe /var/lib/mcp-server/agentos_jwt,
                        0700/0600). Pflicht, weil Agenten- und Dienst-Tokens angenommen werden.
  AUTH_REQUIRE_JWT    — If "true", reject requests without JWT (default: "true", fail-closed)
  MCP_ERLAUBTE_AUD    — OAuth-Zielgruppen (genaue client_ids, kommagetrennt), die das
                        Gateway als oauth_access annimmt (#2976, #3161). Jedes andere aud gibt
                        401, auch bei leerer Liste: dann gelten nur Tokens ohne aud.
  MCP_SUPERADMIN_OHNE_SCOPE — Uebergang (B) bis OBO (#3004): "an" (Vorgabe) liest fuer einen
                        Connector aus MCP_ERLAUBTE_AUD ohne mcp:-Scope is_superadmin aus dem
                        gepruften Token; "aus": Superadmin nur nach dem Pruefmodul.
  REVOKED_AGENT_JTIS  — Notnagel: zusaetzlich gesperrte jti, kommagetrennt.
"""

from __future__ import annotations

import contextvars
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import json as _json

import jwt
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
# Nur fuer gepruefte Tokens mit is_superadmin=true (Betreiber). Ohne diese
# Markierung schraenkt das Gateway Storage-Aufrufe eines Nicht-Agenten ein (#3138).
_current_is_superadmin: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "current_caller_is_superadmin", default=False
)


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


def current_caller_is_superadmin() -> bool:
    """True, wenn das geprüfte JWT des Aufrufers is_superadmin=true trägt."""
    return _current_is_superadmin.get()

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
# JWT-Pruefung: agentos_jwt v1.0.1 (#3161, Teil von #2972/#2970)
# ──────────────────────────────────────────────────────────────
# Vorher stand hier eine eigene Pruefung: JWKS je kid mit Rueckfall auf den einzigen
# Schluessel bei Tokens ohne kid, eigener Abruf der Widerrufsliste (revoked-agent-jtis,
# min_agent_iat) mit eigenem Zustand, aud-Tor in _token_abweisen, verify_aud=False.
# Jetzt das gemeinsame Pruefmodul der auth-api, von Cloud-Security abgenommen (#2987,
# Kommentar 8827): vendort als agentos_jwt.py, Tag jwt-v1.0.1 (225a7e7),
# sha256 ff39755338d7e96d24e2b6a5bdfb6f16e4b6b5f07e7cd541f706f2c6dee2063d
# (tests/test_agentos_jwt_3161.py prueft die Summe). Es prueft:
#   - Signatur nur RS256, Schluessel nur nach kid (ohne kid: abgewiesen), JWKS
#     hoechstens 300 s alt, unbekannter kid laedt einmal nach (hoechstens alle 10 s);
#   - Klasse (type) und aud je Klasse, seltsame Claim-Formen als `malformed` (401);
#   - Widerruf fuer agent/service: jti-Liste, min_agent_iat, owner_min_iat, Zustand in
#     MCP_JWT_STATE_DIR (ein Fehlschlag beim Abruf loescht ihn nie).
# Angenommen am Gateway:
#   agent, service, user  -- ohne aud
#   oauth_access          -- aud aus MCP_ERLAUBTE_AUD (genaue client_ids, #2976)
# Nie: delegated, realtime_grant (Voice), ephemeral_capability, Tokens ohne type.
import agentos_jwt as _aj

# Positive Liste der OAuth-Zielgruppen (client_id), kommagetrennt, je Gateway (#2976,
# Entscheid Cloud 15.11.2026). Seit #3161 GENAUE client_ids: das Modul vergleicht
# exakt, Muster mit * ? [ gelten nicht mehr (Fehlerzeile beim Laden). Jedes andere aud
# gibt 401 -- auch bei LEERER Liste: dann gelten nur Tokens ohne aud (Agenten, Dienste,
# eigenes Login). Das schliesst die Kette ueber anonym per DCR registrierte Clients
# (Befund 76); Tokens fuer andere Anwendungen (OpenCloud-Web, Voice-Grants) fallen
# ebenso heraus.
_ERLAUBTE_AUD = [m.strip() for m in os.getenv("MCP_ERLAUBTE_AUD", "").split(",") if m.strip()]
# Uebergang (B) fuer Connectoren ohne mcp:-Scope, siehe _superadmin().
_SUPERADMIN_OHNE_SCOPE = os.getenv("MCP_SUPERADMIN_OHNE_SCOPE", "an").strip().lower() not in ("aus", "0", "false", "no")
# Widerrufszustand des Moduls (revocation.json, 0600; das Verzeichnis legt das Modul
# mit 0700 an). Unter ProtectSystem=strict liegt es unter /var/lib/mcp-server (#2895).
MCP_JWT_STATE_DIR = os.getenv("MCP_JWT_STATE_DIR", "/var/lib/mcp-server/agentos_jwt")
# Abrufadresse der auth-api fuer JWKS und Widerruf. Auf arkturian (gleicher Host wie
# auth-api): http://localhost:8050. Leer: JWKS aus AUTH_API_JWKS_URL, Widerruf daneben.
MCP_JWT_BASE_URL = os.getenv("MCP_JWT_BASE_URL", "").strip()


def _connector_ids() -> list[str]:
    ids = []
    for m in _ERLAUBTE_AUD:
        if any(z in m for z in "*?["):
            logger.error("MCP_ERLAUBTE_AUD: Muster %r gilt seit #3161 nicht mehr, nur genaue client_ids", m)
            continue
        ids.append(m)
    return ids


def _annahme() -> dict:
    accept: dict = {"agent": None, "service": None, "user": None}
    ids = _connector_ids()
    if ids:
        accept["oauth_access"] = set(ids)
    return accept


def _verifier_bauen(**anders) -> "_aj.Verifier":
    """Der Verifier dieses Prozesses. `anders` nur fuer Tests (ephemeral_state, afetch)."""
    if MCP_JWT_BASE_URL:
        adressen = {"base_url": MCP_JWT_BASE_URL}
    else:
        adressen = {"jwks_url": AUTH_API_JWKS_URL,
                    "revocation_url": AUTH_API_JWKS_URL.rsplit("/", 1)[0] + "/revoked-agent-jtis"}
    args = dict(issuer=AUTH_API_ISSUER, accept=_annahme(), state_dir=MCP_JWT_STATE_DIR, **adressen)
    args.update(anders)
    if args.get("ephemeral_state"):
        args["state_dir"] = None
    return _aj.Verifier(**args)


_verifier_obj: Optional["_aj.Verifier"] = None


def _verifier() -> "_aj.Verifier":
    global _verifier_obj
    if _verifier_obj is None:
        _verifier_obj = _verifier_bauen()
    return _verifier_obj


def _aud_werte(claims: dict) -> list:
    aud = claims.get("aud")
    return [str(a) for a in (aud if isinstance(aud, list) else ([aud] if aud else []))]


def _notnagel_widerrufen(claims: dict) -> bool:
    """REVOKED_AGENT_JTIS (kommagetrennt): Sperre einzelner jti ohne Release, falls die
    Widerrufsliste der auth-api einmal nicht reicht. Zusaetzlich zum Modul, nie statt."""
    jti = str(claims.get("jti") or "")
    if not jti:
        return False
    return jti in {x.strip() for x in os.getenv("REVOKED_AGENT_JTIS", "").split(",") if x.strip()}


def _superadmin(token: str, claims: dict) -> bool:
    """Superadmin am Gateway. Grundsatz: nur der Wert des Moduls -- True allein bei
    user/delegated mit is_superadmin=true; ein Agenten-, Dienst- oder OAuth-Client-Token
    traegt nie Superadmin-Macht.

    UEBERGANG (B), Entscheid Cloud 16.11.2026 (IACP 060afe45): Die Connectoren des
    Betreibers (ChatGPT, iOS-App) tragen keine mcp:-Scopes und kamen bisher nur ueber
    is_superadmin an "*". Damit nichts still wegbricht, liest das Gateway is_superadmin
    aus dem signierten Token -- aber nur fuer oauth_access, dessen aud in
    MCP_ERLAUBTE_AUD steht und das das Modul eben vollstaendig geprueft hat (Signatur,
    Klasse, aud), und nur solange MCP_SUPERADMIN_OHNE_SCOPE an ist. Endet mit OBO
    (#3004): Dann tauscht das Gateway das Connector-Token gegen ein delegated-Token,
    dessen is_superadmin das Modul selbst liefert, und dieser Zweig faellt weg."""
    if claims.get("is_superadmin") is True:
        return True
    if not _SUPERADMIN_OHNE_SCOPE or claims.get("type") != "oauth_access":
        return False
    if not set(_aud_werte(claims)) & set(_connector_ids()):
        return False
    try:
        roh = jwt.decode(token, options={"verify_signature": False})   # dasselbe, eben gepruefte Token
    except jwt.PyJWTError:
        return False
    return isinstance(roh, dict) and roh.get("is_superadmin") is True


async def _pruefen(token: str) -> dict:
    """Die geprueften Claims des Aufrufers. `is_superadmin` ist der Wert am Gateway
    (_superadmin), `type` immer gesetzt. Wirft agentos_jwt.TokenRejected (401),
    RevocationUnknown / KeysUnavailable (503)."""
    claims = await _verifier().averify(token)
    if _notnagel_widerrufen(claims):
        logger.warning("JWT abgewiesen: jti %s per REVOKED_AGENT_JTIS gesperrt (sub=%s)",
                       claims.get("jti"), claims.get("sub"))
        raise _aj.TokenRejected("jti_revoked")
    claims["is_superadmin"] = _superadmin(token, claims)
    return claims


def _rechte_aus_claims(claims: dict) -> list:
    """Die Rechte eines gepruften Tokens -- nie aus dem FEHLEN eines Claims (#2890).

      - Agent und Dienst (type agent/service): genau ihr `permissions`-Claim.
      - Mensch mit OAuth-Token: je Scope `mcp:<server>` das Recht mcp:<server>:*.
      - Superadmin am Gateway (_superadmin: eigenes Login, oder im Uebergang (B) ein
        Connector aus MCP_ERLAUBTE_AUD ohne mcp:-Scope): "*".
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
        return ["*"]
    return []


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
# Die Stufen gelten woertlich, ohne Vererbung -- wie in der person-api selbst --,
# ausser der Server steht in STUFEN_REIHE. server.py meldet die Zuordnung beim
# Einhaengen an.
WERKZEUG_STUFEN: dict[str, dict[str, str]] = {}

# Gestaffelte Stufen (#3130), aufsteigend: eine hoehere Stufe enthaelt alle
# niedrigeren. tree: ("read", "status", "write") wie tree-api app/auth.py
# tree_scopes (write ⊇ status ⊇ read), damit dasselbe Recht ueber REST und MCP
# dasselbe bedeutet. Ohne Eintrag (person) bleiben die Stufen woertlich.
STUFEN_REIHE: dict[str, tuple[str, ...]] = {}


def _stufen_fuer(mcp_server: str, tool_name: str) -> list[str]:
    """Die Stufen, die ein Werkzeug freigeben: seine eigene und, bei gestaffelten
    Servern, jede hoehere. Leer, wenn das Werkzeug keine Stufe hat."""
    stufe = WERKZEUG_STUFEN.get(mcp_server, {}).get(tool_name)
    if stufe is None:
        return []
    reihe = STUFEN_REIHE.get(mcp_server) or ()
    return list(reihe[reihe.index(stufe):]) if stufe in reihe else [stufe]


def _werkzeug_erlaubt(perms: list[str], mcp_server: str, tool_name: str) -> bool:
    """Darf ein Agent mit `perms` (schon nach _permissions_for_server) das Werkzeug rufen?"""
    ziele = [f"mcp:{mcp_server}:{tool_name}"] + [f"mcp:{mcp_server}:{s}" for s in _stufen_fuer(mcp_server, tool_name)]
    return any(_match_permission(p, z) for p in perms for z in ziele)


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
    ziele = [f"mcp:{mcp_server}:{tool_name}"] + [f"mcp:{mcp_server}:{s}" for s in _stufen_fuer(mcp_server, tool_name)]

    def _rang(p: str) -> int:
        # Gleich genau: das Werkzeug selbst vor seiner eigenen Stufe vor hoeheren (#3130).
        return min((i for i, z in enumerate(ziele) if _match_permission(p, z)), default=len(ziele))
    treffer = [p for p in perms if any(_match_permission(p, z) for z in ziele)]
    if not treffer:
        return "?"
    return max(treffer, key=lambda p: (_ap.genauigkeit(p)[0], -_rang(p), _ap.genauigkeit(p)[1]))


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

        # Validate JWT (agentos_jwt, #3161)
        try:
            claims = await _pruefen(token)
        except _aj.TokenRejected as e:
            # Das Modul protokolliert reason, sub und jti selbst, nie das Token.
            return JSONResponse({"error": f"Invalid token: {e.reason}", "reason": e.reason},
                                status_code=401)
        except (_aj.RevocationUnknown, _aj.KeysUnavailable) as e:
            # Schluessel oder Widerrufsstand fehlen (auth-api nicht erreichbar):
            # im strikten Modus 503, sonst wie bisher anonym (Uebergangsmodus).
            logger.error("JWT nicht pruefbar: %s", type(e).__name__)
            if AUTH_REQUIRE_JWT:
                return JSONResponse({"error": "Auth service unavailable",
                                     "reason": type(e).__name__}, status_code=503)
            request.state.agent = AgentContext()
            return await call_next(request)

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
        _superadmin_token = _current_is_superadmin.set(claims.get("is_superadmin") is True)

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
                        required_perm = " | ".join(
                            [f"mcp:{mcp_server}:{tool_name}"]
                            + [f"mcp:{mcp_server}:{s}" for s in _stufen_fuer(mcp_server, tool_name)])
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


        try:
            return await call_next(request)
        finally:
            # Reset ContextVars after request — prevents JWT bleed between calls
            try:
                _current_jwt.reset(_jwt_token)
                _current_agent_name.reset(_name_token)
                _current_is_agent.reset(_is_agent_token)
                _current_is_superadmin.reset(_superadmin_token)
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
