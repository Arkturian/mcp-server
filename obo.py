"""On-behalf-of-Tausch fuer Menschen-Tokens an die Backends (#3004, RFC 8693).

Das Gateway reichte das Token eines Menschen (Login- oder Connector-Token) unveraendert
an content, issue, storage, comm usw. weiter; dort galt ein OAuth-Token fuer den
ChatGPT- oder Claude-Connector wie ein Login-Token, mit Rollen und Superadmin. Jetzt
tauscht das Gateway es bei auth-api gegen ein kurzlebiges Token fuer GENAU einen
Zieldienst (`type=delegated`, `aud=<Ziel>`, Rechte hoechstens die gewaehrten Scopes).

Vertrag: Issue #3004, Kommentare 8421/8422/8430 (auth-api ead0b2c):
    POST {AUTH_API_BASE}/api/v1/auth/oauth/token   (form)
      grant_type=urn:ietf:params:oauth:grant-type:token-exchange
      subject_token=<Token des Menschen>, subject_token_type=...:access_token
      actor_token=<Dienst-JWT des Gateways>, actor_token_type=...:jwt
      audience=<Ziel>
Das Dienst-JWT holt das Gateway per client_credentials (MCP_OBO_CLIENT_ID/_SECRET,
Principal service:mcp-gateway mit oauth:token-exchange).

Schalter (Vorgabe aus, also wie bisher):
    MCP_OBO=1                       Tausch an
    MCP_OBO_ZIELE=cloud-api,...     nur diese Ziele ("*" = alle bekannten)
Agenten- und Dienst-Tokens werden nie getauscht. Scheitert der Tausch, scheitert der
Aufruf -- das rohe Token geht nie als Rueckfall hinaus.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from typing import Optional

import httpx

log = logging.getLogger("mcp-obo")

GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
SUBJECT_TYP = "urn:ietf:params:oauth:token-type:access_token"
AKTEUR_TYP = "urn:ietf:params:oauth:token-type:jwt"
VORLAUF_S = 30                  # getauschtes Token so lange vor exp nicht mehr verwenden
AKTEUR_VORLAUF_S = 60
MAX_EINTRAEGE = 2000

_tausch: dict = {}               # (sha256(subject), audience) -> (token, gueltig_bis)
_akteur: Optional[tuple] = None  # (token, gueltig_bis)
_sperre = asyncio.Lock()


class OboFehler(RuntimeError):
    """Tausch gescheitert; der Aufruf an das Backend unterbleibt."""


def _ziele() -> set:
    return {z.strip() for z in (os.getenv("MCP_OBO_ZIELE") or "").split(",") if z.strip()}


def aktiv(audience: str) -> bool:
    if (os.getenv("MCP_OBO") or "").strip() != "1" or not audience:
        return False
    z = _ziele()
    return "*" in z or audience in z


def _auth_base() -> str:
    return (os.getenv("AUTH_API_BASE") or "").strip().rstrip("/")


async def _post(daten: dict) -> dict:
    base = _auth_base()
    if not base:
        raise OboFehler("obo_nicht_konfiguriert: AUTH_API_BASE fehlt")
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.post(f"{base}/api/v1/auth/oauth/token", data=daten)
    try:
        js = r.json()
    except ValueError:
        js = {}
    if r.status_code != 200 or not isinstance(js, dict) or not js.get("access_token"):
        fehler = (js or {}).get("error") if isinstance(js, dict) else None
        raise OboFehler(f"obo_{fehler or 'fehler'} (HTTP {r.status_code})")
    return js


async def _akteur_token() -> str:
    global _akteur
    jetzt = time.time()
    if _akteur and _akteur[1] > jetzt:
        return _akteur[0]
    cid = (os.getenv("MCP_OBO_CLIENT_ID") or "").strip()
    geheim = (os.getenv("MCP_OBO_CLIENT_SECRET") or "").strip()
    if not cid or not geheim:
        raise OboFehler("obo_nicht_konfiguriert: MCP_OBO_CLIENT_ID/_SECRET fehlen")
    js = await _post({"grant_type": "client_credentials", "client_id": cid, "client_secret": geheim})
    dauer = int(js.get("expires_in") or 3600)
    _akteur = (js["access_token"], jetzt + max(0, dauer - AKTEUR_VORLAUF_S))
    return _akteur[0]


async def tausche(subject_token: str, audience: str) -> str:
    """Kurzlebiges Token fuer `audience` an Stelle von `subject_token`."""
    if not subject_token:
        raise OboFehler("obo_ohne_aufrufer")
    schluessel = (hashlib.sha256(subject_token.encode()).hexdigest(), audience)
    jetzt = time.time()
    t = _tausch.get(schluessel)
    if t and t[1] > jetzt:
        return t[0]
    async with _sperre:
        t = _tausch.get(schluessel)
        if t and t[1] > time.time():
            return t[0]
        akteur = await _akteur_token()
        js = await _post({"grant_type": GRANT, "subject_token": subject_token,
                          "subject_token_type": SUBJECT_TYP, "actor_token": akteur,
                          "actor_token_type": AKTEUR_TYP, "audience": audience})
        dauer = int(js.get("expires_in") or 0)
        if len(_tausch) >= MAX_EINTRAEGE:
            for k in [k for k, v in _tausch.items() if v[1] <= time.time()] or list(_tausch)[:MAX_EINTRAEGE // 2]:
                _tausch.pop(k, None)
        _tausch[schluessel] = (js["access_token"], time.time() + max(0, dauer - VORLAUF_S))
        log.info("obo: getauscht fuer %s (gueltig %ss)", audience, dauer)
        return js["access_token"]


def _reset_for_tests() -> None:
    global _akteur
    _akteur = None
    _tausch.clear()
