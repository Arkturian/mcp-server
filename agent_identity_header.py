"""Agentenname im HTTP-Kopf — auch mit Umlaut (Issue #2402, ELE 12.10.2026).

HTTP-Kopfwerte sind ASCII; httpx wirft bei "ü" einen UnicodeEncodeError, bevor
die Anfrage hinausgeht. Ein Agent mit Nicht-ASCII-Zeichen im Namen konnte so
kein einziges Gateway-Werkzeug aufrufen. Sender und Empfaenger teilen deshalb
diese zwei Funktionen (Vorschlag des Cloud-Agenten der ELE-Instanz):

    ASCII-Name      -> Kopf wie bisher, byteidentisch, kein Zusatzkopf
    sonstiger Name  -> prozentkodiert (UTF-8) + "X-Agent-Name-Encoding: pct"

Dekodiert wird NUR mit Markierung: ein ASCII-Name, der selbst ein "%" traegt,
bleibt, wie er ist. Eine kaputte Kodierung ergibt "" (kein Name, kein Zutritt).
Die Markierung ist so vertrauenswuerdig wie der Namenskopf selbst — wer dem
Kopf nur mit Gateway-Beweis (X-IACP-Token) glaubt, tut das hier weiterhin.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote, unquote

NAME_HEADER = "X-Agent-Name"
ENCODING_HEADER = "X-Agent-Name-Encoding"
PCT = "pct"


def set_agent_name(headers: dict, name: str) -> dict:
    """Namen in `headers` eintragen; gibt `headers` zurueck."""
    if not name:
        return headers
    if name.isascii():
        headers[NAME_HEADER] = name
        headers.pop(ENCODING_HEADER, None)
    else:
        headers[NAME_HEADER] = quote(name, safe="")
        headers[ENCODING_HEADER] = PCT
    return headers


def get_agent_name(headers: Any) -> str:
    """Namen aus den Koepfen lesen (dekodiert, wenn markiert)."""
    raw = str(headers.get(NAME_HEADER) or "").strip()
    if not raw:
        return ""
    if str(headers.get(ENCODING_HEADER) or "").strip().lower() != PCT:
        return raw
    try:
        return unquote(raw, encoding="utf-8", errors="strict").strip()
    except (UnicodeDecodeError, ValueError):
        return ""
