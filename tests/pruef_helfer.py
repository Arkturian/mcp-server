"""Hilfen fuer Tests hinter der JWT-Pruefung (#3161).

Seit agentos_jwt prueft das Gateway Tokens in `auth._pruefen` (async). Tests, die nur
das RBAC dahinter brauchen, ersetzen diese Stelle durch feste Claims:

    monkeypatch.setattr(auth, "_pruefen", claims_fest(lambda token: {...}))

Die Pruefung selbst (Signatur, kid, Klasse, aud, Widerruf) testet
tests/test_agentos_jwt_3161.py mit echt signierten Tokens.
"""


def claims_fest(fn):
    async def _pruefen(token):
        return fn(token)
    return _pruefen
