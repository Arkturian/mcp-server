"""Post 5782: transcribe_audio liest private Dateien im Auftrag des Aufrufers.

Vorher schickte das Werkzeug JSON an eine Formular-Route und bekam immer 422.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

SIGNIERT = "https://storage.example/storage/media/7?exp=1&kid=k1&sig=geheim"
DIARIZED = {"text": "Hallo Servus Ja", "model": "gpt-4o-transcribe-diarize", "diarized": {"segments": [
    {"speaker": "A", "start": 1.0, "text": "Hallo"},
    {"speaker": "A", "start": 2.0, "text": "Servus"},
    {"speaker": "B", "start": 65.0, "text": "Ja"},
]}}


@pytest.fixture
def als_agent():
    import auth
    t = auth._current_agent_name.set("Business"), auth._current_is_agent.set(True)
    yield
    auth._current_agent_name.reset(t[0])
    auth._current_is_agent.reset(t[1])


@pytest.fixture
def netz(monkeypatch):
    import server
    calls = []

    async def fetch(method, url, **kw):
        calls.append((url, kw))
        if "media-urls" in url:
            return {"items": [{"id": 7, "url": SIGNIERT}]}
        if "/ai/transcribe" in url:
            return DIARIZED
        raise AssertionError(url)

    monkeypatch.setattr(server, "_fetch_json", fetch)
    return calls


def _aufruf(**kw):
    import server
    return asyncio.run(server.ai_transcribe_audio(**kw))


def test_storage_id_wird_signiert_und_als_formular_geschickt(als_agent, netz):
    r = _aufruf(storage_id=7, speakers=True)
    url, kw = netz[-1]
    assert kw["form"] == {"file_url": SIGNIERT, "model": "gpt-4o-transcribe-diarize",
                          "response_format": "diarized_json"}
    assert not kw.get("json_body") and not kw.get("params")
    assert kw["timeout"] >= 1200
    assert r == DIARIZED


def test_signatur_steht_nie_im_ergebnis(als_agent, netz, monkeypatch):
    import server

    async def upload(*a, **k):
        return {"id": 9}

    monkeypatch.setattr(server, "call_storage_upload", upload)
    assert "sig=" not in str(_aufruf(storage_id=7, speakers=True))
    assert "sig=" not in str(_aufruf(storage_id=7, speakers=True, save_result=True))


def test_ohne_bindung_kein_aufruf_bei_aiapi(als_agent, monkeypatch):
    import server
    calls = []

    async def fetch(method, url, **kw):
        calls.append(url)
        return {"items": [{"id": 7, "error": "not_granted"}]}

    monkeypatch.setattr(server, "_fetch_json", fetch)
    assert _aufruf(storage_id=7) == {"id": 7, "error": "not_granted"}
    assert not [u for u in calls if "/ai/transcribe" in u]


def test_storage_id_verlangt_einen_agenten(netz):
    with pytest.raises(Exception, match="requires_agent"):
        _aufruf(storage_id=7)
    assert netz == []


def test_oeffentliche_adresse_bleibt_moeglich(netz):
    _aufruf(file_url="https://example.com/a.mp3", model="whisper-1", language="sl")
    assert netz[-1][1]["form"] == {"file_url": "https://example.com/a.mp3", "model": "whisper-1", "language": "sl"}


@pytest.mark.parametrize("kw", [{}, {"storage_id": 7, "file_url": "https://example.com/a.mp3"}])
def test_genau_eine_quelle(kw):
    with pytest.raises(ValueError):
        _aufruf(**kw)


def test_speakers_und_modell_schliessen_sich_aus():
    with pytest.raises(ValueError):
        _aufruf(file_url="https://example.com/a.mp3", speakers=True, model="whisper-1")


def test_ablage_ist_privat_an_den_aufrufer_gebunden(als_agent, netz, monkeypatch):
    import server
    seen = {}

    async def upload(data, name, **k):
        seen.update(data=data, name=name, **k)
        return {"id": 9}

    monkeypatch.setattr(server, "call_storage_upload", upload)
    r = _aufruf(storage_id=7, speakers=True, save_result=True)
    assert seen["form_fields"]["private"] == "true" and seen["form_fields"]["is_public"] == "false"
    assert not seen.get("share_scopes")
    assert seen["data"].decode() == "[00:00:01] Sprecher A: Hallo Servus\n\n[00:01:05] Sprecher B: Ja\n"
    assert r["storage_id"] == 9 and r["speakers"] == ["A", "B"]
    assert "diarized" not in r


def test_ablage_nur_fuer_agenten(netz):
    with pytest.raises(Exception, match="requires_agent"):
        _aufruf(file_url="https://example.com/a.mp3", save_result=True)
