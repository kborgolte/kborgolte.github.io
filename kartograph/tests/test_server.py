"""Tests für die HTTP-Schnittstelle.

Prüft das Verhalten, das man beim Ausprobieren am schwersten selbst nachstellt:
abgelehnte Dateitypen, fehlender Checkpoint, Löschen laufender Jobs. Der Worker
läuft dabei nicht — es geht um die Schnittstelle, nicht um die Rekonstruktion.

Aufruf:  python tests/test_server.py
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Datenverzeichnis umbiegen, bevor irgendetwas aus kartograph importiert wird —
# sonst schreiben die Tests in die echten Szenen.
_TMP = tempfile.mkdtemp(prefix="kartograph-test-")
import os  # noqa: E402

os.environ["KARTOGRAPH_DATA"] = _TMP

from fastapi.testclient import TestClient  # noqa: E402

from kartograph import config  # noqa: E402
from kartograph.server import app, store  # noqa: E402


def client() -> TestClient:
    return TestClient(app)


def test_status_survives_missing_torch() -> None:
    """Die Statusseite muss auch bei unvollständiger Installation antworten."""
    with client() as web:
        response = web.get("/api/status")

    assert response.status_code == 200, f"Status {response.status_code}"
    payload = response.json()
    assert "device" in payload and "label" in payload["device"]
    assert isinstance(payload["model_ready"], bool)
    assert payload["mobile_url"].startswith("http://")
    print(f"  ✓ /api/status antwortet ({payload['device']['label']})")


def test_pages_render() -> None:
    with client() as web:
        for path, needle in [("/", "Kartograph"), ("/m", "Video aufnehmen")]:
            response = web.get(path)
            assert response.status_code == 200, f"{path} → {response.status_code}"
            assert needle in response.text, f"{needle!r} fehlt in {path}"
    print("  ✓ Notebook- und Handy-Seite werden ausgeliefert")


def test_upload_rejects_non_video() -> None:
    """Eine PDF darf gar nicht erst in der Warteschlange landen."""
    with client() as web:
        response = web.post(
            "/api/upload",
            files={"file": ("notiz.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

    assert response.status_code == 400, f"Status {response.status_code} statt 400"
    assert ".pdf" in response.json()["detail"]
    print("  ✓ Falscher Dateityp wird mit Begründung abgelehnt")


def test_upload_reports_missing_model() -> None:
    """Ohne Checkpoint muss die Meldung auf setup.sh zeigen, nicht auf einen Stacktrace."""
    with client() as web:
        response = web.post(
            "/api/upload",
            files={"file": ("clip.mp4", io.BytesIO(b"\x00" * 128), "video/mp4")},
        )

    if config.MODEL_PATH.exists():
        print("  – Checkpoint vorhanden, Test übersprungen")
        return

    assert response.status_code == 503, f"Status {response.status_code} statt 503"
    assert "setup.sh" in response.json()["detail"]
    print("  ✓ Fehlender Checkpoint wird verständlich gemeldet")


def test_unknown_job_is_404() -> None:
    with client() as web:
        for path in ["/api/jobs/gibtsnicht", "/api/scene/gibtsnicht/points.bin"]:
            assert web.get(path).status_code == 404, f"{path} war kein 404"
    print("  ✓ Unbekannte Jobs liefern 404")


def test_running_job_cannot_be_deleted() -> None:
    """Ein laufender Job darf nicht unter dem Worker weggelöscht werden."""
    from kartograph.jobs import Job

    with client() as web:
        # Direkt eintragen statt über create(): so landet der Job nicht in der
        # Warteschlange und der Worker fasst ihn nicht an. Der Aufbau muss
        # innerhalb des Kontexts passieren, weil der Lifespan beim Start
        # load_from_disk() aufruft und dabei alles Vorherige überschreibt.
        video = Path(_TMP) / "laeuft.mp4"
        video.write_bytes(b"\x00" * 32)

        job = store.add(Job(id="testlaeuft", name="laeuft.mp4", video_path=str(video)))
        job.status = "running"
        job_id = job.id

        response = web.delete(f"/api/jobs/{job_id}")
        assert response.status_code == 409, f"Status {response.status_code} statt 409"

        # Abbrechen ist erlaubt, danach das Löschen auch.
        assert web.post(f"/api/jobs/{job_id}/cancel").status_code == 200
        assert web.delete(f"/api/jobs/{job_id}").status_code == 200

    print("  ✓ Laufende Jobs erst abbrechen, dann löschen")


def test_job_list_shape() -> None:
    with client() as web:
        payload = web.get("/api/jobs").json()

    assert isinstance(payload.get("jobs"), list)
    print("  ✓ Jobliste hat die erwartete Form")


if __name__ == "__main__":
    print("\nServer-Tests\n")
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"\n{len(tests)} Tests bestanden.\n")
