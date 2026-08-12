"""Webserver: Upload vom Handy, Auftragsverwaltung, Auslieferung der Szenen.

Der Server bindet absichtlich an 0.0.0.0, damit das Handy im selben WLAN
draufkommt. Es gibt keine Anmeldung — das ist eine App fürs eigene Netz, nicht
fürs offene Internet. Wer sie nach außen stellen will, gehört hinter einen
Reverse Proxy mit Authentifizierung.
"""

from __future__ import annotations

import asyncio
import shutil
import socket
import time
import uuid
from contextlib import asynccontextmanager, closing
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from kartograph import config, pipeline
from kartograph.jobs import CANCELLED, TERMINAL, JobStore

STATIC_DIR = Path(__file__).parent / "static"

store = JobStore()
cancels: dict[str, asyncio.Event] = {}


def lan_ip() -> str:
    """Die IP finden, unter der das Handy den Rechner erreicht.

    Der UDP-Socket verschickt nichts; er zwingt das Betriebssystem nur, eine
    Route zu wählen, und verrät damit die richtige lokale Adresse. Zuverlässiger
    als gethostbyname, das auf macOS gern 127.0.0.1 liefert.
    """
    try:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
            sock.connect(("192.0.2.1", 1))  # TEST-NET-1, garantiert unerreichbar
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def mobile_url() -> str:
    return f"http://{lan_ip()}:{config.PORT}/m"


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    store.load_from_disk()
    task = asyncio.create_task(pipeline.worker(store, cancels))

    url = mobile_url()
    print(f"\n  {config.APP_NAME} läuft")
    print(f"  Notebook:  http://localhost:{config.PORT}")
    print(f"  Handy:     {url}\n")

    yield

    task.cancel()


app = FastAPI(title=config.APP_NAME, lifespan=lifespan)


# ── Seiten ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/m", response_class=HTMLResponse)
async def mobile() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "mobile.html").read_text(encoding="utf-8"))


# ── Status und QR-Code ───────────────────────────────────────────────────────

@app.get("/api/status")
async def status() -> dict:
    try:
        # Erst hier importieren: der Import zieht torch nach, und die
        # Statusseite soll gerade dann etwas Sinnvolles sagen können, wenn die
        # Installation noch unvollständig ist.
        from kartograph.device import pick_device

        choice = pick_device().as_dict()
    except Exception as exc:  # noqa: BLE001 - Statusabfrage darf nie 500en
        choice = {"device": "none", "label": "PyTorch fehlt", "note": str(exc)}

    return {
        "app": config.APP_NAME,
        "device": choice,
        "model_ready": config.MODEL_PATH.exists(),
        "model_path": str(config.MODEL_PATH),
        "lingbot_ready": config.LINGBOT_REPO.exists(),
        "mobile_url": mobile_url(),
        "queue_depth": store.queue.qsize(),
    }


@app.get("/api/qr")
async def qr_code() -> Response:
    """QR-Code auf die Handy-Seite, damit man am Handy nichts tippen muss."""
    url = mobile_url()
    try:
        import qrcode
        import qrcode.image.svg

        image = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage)
        buffer = image.to_string()
        return Response(buffer, media_type="image/svg+xml")
    except ImportError:
        raise HTTPException(501, "qrcode ist nicht installiert — URL bitte abtippen")


# ── Upload ───────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload(file: UploadFile) -> JSONResponse:
    name = Path(file.filename or "video.mp4").name
    suffix = Path(name).suffix.lower()

    if suffix not in config.VIDEO_SUFFIXES:
        raise HTTPException(
            400,
            f"'{suffix or 'ohne Endung'}' wird nicht unterstützt. "
            f"Erlaubt: {', '.join(sorted(config.VIDEO_SUFFIXES))}",
        )

    if not config.MODEL_PATH.exists():
        raise HTTPException(503, "Der Modell-Checkpoint fehlt. Bitte ./setup.sh ausführen.")

    config.ensure_dirs()
    # Zeitstempel und Zufallssuffix, damit zwei Videos gleichen Namens sich
    # nicht überschreiben — bei Handys heißt fast alles IMG_0001.mov.
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    target = config.UPLOAD_DIR / f"{stamp}_{name}"

    written = 0
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        f"Video größer als {config.MAX_UPLOAD_BYTES // 1024**3} GB. "
                        "Bitte kürzer aufnehmen.",
                    )
                handle.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise

    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(400, "Die Datei war leer.")

    job = await store.create(name=name, video_path=target)
    return JSONResponse({"job": job.to_dict()}, status_code=201)


# ── Jobs ─────────────────────────────────────────────────────────────────────

@app.get("/api/jobs")
async def list_jobs() -> dict:
    return {"jobs": [job.to_dict() for job in store.all()]}


@app.get("/api/jobs/{job_id}")
async def job_detail(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Unbekannter Job")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Unbekannter Job")
    if job.status in TERMINAL:
        raise HTTPException(409, f"Job ist bereits {job.status}")

    event = cancels.get(job_id)
    if event is not None:
        event.set()
    else:
        # Noch nicht gestartet: direkt als abgebrochen markieren, der Worker
        # überspringt ihn dann beim Herausnehmen aus der Warteschlange.
        job.status = CANCELLED
        job.stage = "Abgebrochen"
        store.persist(job)

    return job.to_dict()


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Unbekannter Job")
    if job.status not in TERMINAL:
        raise HTTPException(409, "Laufende Jobs erst abbrechen")

    shutil.rmtree(job.scene_dir, ignore_errors=True)
    Path(job.video_path).unlink(missing_ok=True)
    store.remove(job_id)
    return {"deleted": job_id}


# ── Szenendaten ──────────────────────────────────────────────────────────────

def _scene_file(job_id: str, filename: str) -> Path:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Unbekannter Job")
    path = job.scene_dir / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} gibt es für diesen Job (noch) nicht")
    return path


@app.get("/api/scene/{job_id}/points.bin")
async def scene_points(job_id: str) -> FileResponse:
    return FileResponse(_scene_file(job_id, "points.bin"), media_type="application/octet-stream")


@app.get("/api/scene/{job_id}/scene.glb")
async def scene_glb(job_id: str) -> FileResponse:
    job = store.get(job_id)
    filename = f"{Path(job.name).stem if job else job_id}.glb"
    return FileResponse(
        _scene_file(job_id, "scene.glb"),
        media_type="model/gltf-binary",
        filename=filename,
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
