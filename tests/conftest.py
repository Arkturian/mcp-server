"""Gemeinsame Vorkehrungen fuer die Tests des Gateways."""
import pytest


@pytest.fixture(autouse=True)
def _aufruf_protokoll_im_tmp(tmp_path, monkeypatch):
    """Das Aufruf-Protokoll (#2806) schreibt nie in den echten Pfad
    (/var/lib/mcp-server/aufrufe.db): jeder Test bekommt eine eigene Datei, auch
    wenn er den Lebenszyklus der App (Start, Beenden -> schreiben) durchlaeuft."""
    import aufruf_protokoll
    monkeypatch.setattr(aufruf_protokoll, "_instanz", aufruf_protokoll.Protokoll(str(tmp_path / "aufrufe.db")))


@pytest.fixture(autouse=True)
def _live_rechte_ohne_netz(monkeypatch):
    """#3987: Der Live-Abgleich fragte sonst in jedem Test mit business-Aufruf die echte auth-api
    (gesehen: HTTP 401 mit dem Test-Token). Hier aus; test_business_live_rechte_3987 schaltet ihn
    mit einer Attrappe ein. Ein vergessener Aufruf scheitert laut statt ins Netz zu gehen."""
    import auth
    monkeypatch.setattr(auth, "LIVE_RECHTE_SERVER", frozenset())
    monkeypatch.setattr(auth, "_LIVE_RECHTE", {})

    async def _kein_netz(name, token):
        raise AssertionError("Live-Rechte-Abruf ohne Attrappe (#3987)")
    monkeypatch.setattr(auth, "_live_rechte_abruf", _kein_netz)
