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
- GET-Stroeme (#4180, config 27.11.: 100 offene Stroeme belegten den ganzen
  Verbindungsvorrat, jeder POST endete in PoolTimeout): eigener Client je Ziel,
  hoechstens MCP_PASSTHROUGH_STROEME (Vorgabe 32) zugleich. Darueber bekommt ein
  GET sofort 405; nach dem Streamable-HTTP-Transport heisst das "kein Strom vom
  Server", der MCP-Client arbeitet ohne weiter. POST und DELETE haben ihren
  eigenen Vorrat und scheitern nie an offenen Stroemen.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Dict, Optional
from urllib.parse import urlsplit

import httpx
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


STROEME_VORGABE = 32
GRENZE_MELDEN_S = 60.0


def stroeme_grenze(roh: Optional[str] = None) -> int:
    """Hoechstzahl gleichzeitiger GET-Stroeme je Ziel (MCP_PASSTHROUGH_STROEME). 0 = keine Stroeme."""
    roh = os.getenv("MCP_PASSTHROUGH_STROEME") if roh is None else roh
    try:
        return max(0, int(str(roh).strip())) if str(roh or "").strip() else STROEME_VORGABE
    except ValueError:
        logger.warning("MCP_PASSTHROUGH_STROEME ungueltig (%r), Vorgabe %d", str(roh)[:20], STROEME_VORGABE)
        return STROEME_VORGABE


def _ziel(basis: str, pfad: str, query: str) -> str:
    rest = pfad.lstrip("/")
    url = f"{basis}/{rest}" if rest else basis
    return f"{url}?{query}" if query else url


class Durchreiche:
    """ASGI-App fuer genau einen Zieldienst."""

    def __init__(self, name: str, basis: str, transport: Optional[httpx.AsyncBaseTransport] = None,
                 stroeme: Optional[int] = None):
        self.name = name
        self.basis = basis
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None          # POST, DELETE
        self._strom_client: Optional[httpx.AsyncClient] = None    # GET-Stroeme (#4180)
        self.stroeme = stroeme_grenze() if stroeme is None else max(0, int(stroeme))
        self.offene_stroeme = 0
        self._gemeldet = 0.0

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=10.0),
                follow_redirects=False,
            )
        return self._client

    def _http_strom(self) -> httpx.AsyncClient:
        """Eigener Vorrat fuer GET-Stroeme: sie halten ihre Verbindung beliebig lange und duerfen den der
        Aufrufe nie belegen. Die Zahl begrenzt __call__ selbst (offene_stroeme), der Vorrat passt dazu."""
        if self._strom_client is None:
            self._strom_client = httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0),
                limits=httpx.Limits(max_connections=max(1, self.stroeme), max_keepalive_connections=0),
                follow_redirects=False,
            )
        return self._strom_client

    async def schliessen(self) -> None:
        for c in (self._client, self._strom_client):
            if c is not None:
                await c.aclose()
        self._client = self._strom_client = None

    def _grenze_melden(self) -> None:
        jetzt = time.monotonic()
        if jetzt - self._gemeldet >= GRENZE_MELDEN_S:
            self._gemeldet = jetzt
            logger.warning("Durchreiche %s: Grenze fuer GET-Stroeme erreicht (%d offen, MCP_PASSTHROUGH_STROEME), "
                           "weitere GET bekommen 405", self.name, self.offene_stroeme)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        request = Request(scope, receive)
        # Neuere Starlette laesst unter Mount den vollen Pfad stehen und setzt
        # root_path auf das Praefix; aeltere kuerzen den Pfad selbst.
        pfad, praefix = scope.get("path", ""), scope.get("root_path", "")
        if praefix and pfad.startswith(praefix):
            pfad = pfad[len(praefix):]
        # Nur der MCP-Endpunkt selbst (#2893): kein Unterpfad, nur die Verben
        # des Streamable-HTTP-Transports. Sonst reichte das Gateway jeden Pfad
        # und jede Methode an den internen Dienst weiter.
        if pfad.strip("/"):
            await JSONResponse({"error": "Not Found"}, status_code=404)(scope, receive, send)
            return
        if request.method not in ("POST", "GET", "DELETE"):
            await JSONResponse({"error": "Method Not Allowed"}, status_code=405)(scope, receive, send)
            return
        strom = request.method == "GET"
        if strom and self.offene_stroeme >= self.stroeme:
            # #4180: kein weiterer Strom; 405 heisst fuer den MCP-Client "der Server bietet keinen".
            self._grenze_melden()
            await JSONResponse({"error": "Method Not Allowed", "grund": "strom_grenze"}, status_code=405,
                               headers={"Allow": "POST, DELETE"})(scope, receive, send)
            return
        ziel = _ziel(self.basis, pfad, scope.get("query_string", b"").decode("latin-1"))
        koepfe = [(k, v) for k, v in request.headers.items() if k.lower() not in _NICHT_HIN]
        body = await request.body()
        client = self._http_strom() if strom else self._http()
        if strom:
            self.offene_stroeme += 1
        try:
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
            try:
                await StreamingResponse(antwort.aiter_raw(), status_code=status, headers=zurueck)(
                    scope, receive, send)
            finally:
                await antwort.aclose()           # auch beim Abbruch des Aufrufers (Verbindung zurueck)
        finally:
            if strom:
                self.offene_stroeme -= 1


def aus_umgebung(reserved=()) -> Dict[str, Durchreiche]:
    return {n: Durchreiche(n, u) for n, u in parse(os.getenv("MCP_PASSTHROUGH"), reserved).items()}
