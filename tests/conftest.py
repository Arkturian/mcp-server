"""Gemeinsame Vorkehrungen fuer die Tests des Gateways."""
import pytest


@pytest.fixture(autouse=True)
def _aufruf_protokoll_im_tmp(tmp_path, monkeypatch):
    """Das Aufruf-Protokoll (#2806) schreibt nie in den echten Pfad
    (/var/lib/mcp-server/aufrufe.db): jeder Test bekommt eine eigene Datei, auch
    wenn er den Lebenszyklus der App (Start, Beenden -> schreiben) durchlaeuft."""
    import aufruf_protokoll
    monkeypatch.setattr(aufruf_protokoll, "_instanz", aufruf_protokoll.Protokoll(str(tmp_path / "aufrufe.db")))
