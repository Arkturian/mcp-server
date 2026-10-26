"""MCP-Durchreiche: fremde MCP-Server hinter dem Gateway (P3 26.10.2026).

Anlass: P3 betreibt einen eigenen, nur lesenden MCP-Dienst (p3-buchhaltung,
127.0.0.1:8031/mcp in der VM). Die Agenten laufen auf dem Mac und erreichen
die VM nur ueber die Lima-Freigabeliste; der Gateway ist schon dort. Ein
eigener Eintrag je Agent (local_mcp) bekaeme absichtlich kein JWT, ein
Eintrag in der Vorlage stuende an jedem Agenten.

Konfiguration ueber die Umgebung, kommagetrennt:

    MCP_PASSTHROUGH=p3-buchhaltung=http://127.0.0.1:8031/mcp

Regeln:
- Gemountet unter /<name>/ wie jeder andere Server. Damit gilt die Pruefung
  des JWTAuthMiddleware unveraendert: tools/call braucht ein gueltiges JWT
  und `mcp:<name>:*` (bzw. das Werkzeugrecht), und prepare_bot_home zeigt
  den Server nur Agenten mit diesem Recht.
- Der Authorization-Kopf des Agenten geht UNVERAENDERT durch. Der Gateway
  setzt keine eigene Identitaet und kein Dienst-Token; der Zieldienst prueft
  das JWT selbst.
- Nicht durchgereicht: Host, Hop-by-Hop-Koepfe, Cookies und das IACP-Token.
- Antworten werden gestreamt (SSE bei streamable HTTP), ohne Lese-Timeout.
- 401 des Zieldienstes wird 403, `WWW-Authenticate` faellt weg: Claude Code
  startet bei 401 mit resource_metadata seinen OAuth-Ablauf, der Agent haengt
  dann am Anmeldebanner. Der Gateway selbst antwortet aus demselben Grund 403.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, Optional
from urllib.parse import urlsplit

import httpx
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("mcp-passthrough")

NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,39}$")

# RFC 7230 §6.1 plus Koepfe, die nie an einen fremden Dienst gehen
_NICHT_HIN = {
    "host", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
    "cookie", "x-iacp-token",
}
_NICHT_ZURUECK = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length", "set-cookie",
    "www-authenticate",
}


def parse(raw: Optional[str], reserved=()) -> Dict[str, str]:
    """`name=url,name=url` -> {name: url}. Ungueltiges wird geloggt und ausgelassen."""
    out: Dict[str, str] = {}
    for teil in (raw or "").split(","):
        teil = teil.strip()
        if not teil:
            continue
        name, _, url = teil.partition("=")
        name, url = name.strip().lower(), url.strip()
        try:
            u = urlsplit(url)
            gueltig_url = u.scheme in ("http", "https") and bool(u.hostname) \
                and not u.username and not u.password
        except ValueError:
            gueltig_url = False
        if not NAME_RE.match(name) or not gueltig_url:
            logger.warning("MCP_PASSTHROUGH: Eintrag verworfen: %r", teil[:80])
            continue
        if name in reserved or name in out:
            logger.warning("MCP_PASSTHROUGH: Name %r schon vergeben, verworfen", name)
            continue
        out[name] = url.rstrip("/")
    return out


def _ziel(basis: str, pfad: str, query: str) -> str:
    rest = pfad.lstrip("/")
    url = f"{basis}/{rest}" if rest else basis
    return f"{url}?{query}" if query else url


class Durchreiche:
    """ASGI-App fuer genau einen Zieldienst."""

    def __init__(self, name: str, basis: str, transport: Optional[httpx.AsyncBaseTransport] = None):
        self.name = name
        self.basis = basis
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=10.0),
                follow_redirects=False,
            )
        return self._client

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        request = Request(scope, receive)
        # Neuere Starlette laesst unter Mount den vollen Pfad stehen und setzt
        # root_path auf das Praefix; aeltere kuerzen den Pfad selbst.
        pfad, praefix = scope.get("path", ""), scope.get("root_path", "")
        if praefix and pfad.startswith(praefix):
            pfad = pfad[len(praefix):]
        ziel = _ziel(self.basis, pfad, scope.get("query_string", b"").decode("latin-1"))
        koepfe = [(k, v) for k, v in request.headers.items() if k.lower() not in _NICHT_HIN]
        body = await request.body()
        client = self._http()
        try:
            antwort = await client.send(
                client.build_request(request.method, ziel, headers=koepfe, content=body),
                stream=True,
            )
        except httpx.HTTPError as e:
            logger.warning("Durchreiche %s: Ziel nicht erreichbar: %r", self.name, e)
            await JSONResponse({"error": f"{self.name}: Zieldienst nicht erreichbar"},
                               status_code=502)(scope, receive, send)
            return
        zurueck = {k: v for k, v in antwort.headers.items() if k.lower() not in _NICHT_ZURUECK}
        status = 403 if antwort.status_code == 401 else antwort.status_code
        await StreamingResponse(
            antwort.aiter_raw(), status_code=status, headers=zurueck,
            background=BackgroundTask(antwort.aclose),
        )(scope, receive, send)


def aus_umgebung(reserved=()) -> Dict[str, Durchreiche]:
    return {n: Durchreiche(n, u) for n, u in parse(os.getenv("MCP_PASSTHROUGH"), reserved).items()}
