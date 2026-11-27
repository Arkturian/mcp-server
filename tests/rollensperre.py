"""#4171: Eine Sperre nach Werkzeugrecht bei gueltigem Token kommt als Werkzeugergebnis (HTTP 200,
isError, Text beginnt mit `rolle_gesperrt:`), nicht mehr als HTTP 403. Ein Stapel und die Sperre auf
Server-Ebene bleiben bei 403. Hilfsmittel fuer die Tests, die bisher nur den Statuscode lasen."""

GESPERRT = "rolle_gesperrt"


def code(r):
    """Statuscode der Antwort, oder GESPERRT fuer eine Sperre nach Rolle als Werkzeugergebnis."""
    if r.status_code == 200:
        try:
            j = r.json()
        except Exception:
            return 200
        res = j.get("result") if isinstance(j, dict) else None
        if isinstance(res, dict) and res.get("isError"):
            inhalt = res.get("content") or [{}]
            text = str((inhalt[0] if isinstance(inhalt[0], dict) else {}).get("text") or "")
            if text.startswith("rolle_gesperrt:"):
                return GESPERRT
    return r.status_code
