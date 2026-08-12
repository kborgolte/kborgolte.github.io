"""Der Arbeiter, der Jobs der Reihe nach abarbeitet.

Genau ein Job zur Zeit. Zwei parallele Rekonstruktionen würden sich auf einer
GPU gegenseitig den Speicher nehmen und beide langsamer machen als sie
nacheinander wären.

Zwei Schritte pro Job:
  1. runner_inference.py  Video → predictions.npz   (dauert, braucht die GPU)
  2. export_scene.py      NPZ   → points.bin + GLB  (schnell, reines NumPy)

Beide laufen als Unterprozess und melden Fortschritt über ``KG_PROGRESS``-Zeilen.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from kartograph import config
from kartograph.jobs import CANCELLED, DONE, FAILED, QUEUED, RUNNING, Job, JobStore


async def _stream_process(
    args: list[str], job: Job, store: JobStore, cancel: asyncio.Event
) -> None:
    """Unterprozess starten und seine Ausgabe live in den Job spiegeln."""
    env = {
        **os.environ,
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "PYTHONUNBUFFERED": "1",
        # Ohne das findet der Unterprozess das kartograph-Paket nicht, wenn die
        # App aus einem anderen Arbeitsverzeichnis gestartet wurde.
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(config.PROJECT_ROOT), os.environ.get("PYTHONPATH", "")])
        ),
    }

    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        cwd=str(config.PROJECT_ROOT),
    )

    async def watch_for_cancel() -> None:
        await cancel.wait()
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                process.kill()

    canceller = asyncio.create_task(watch_for_cancel())

    try:
        assert process.stdout is not None
        async for raw in process.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue

            if line.startswith("KG_PROGRESS "):
                _, percent, message = line.split(" ", 2)
                job.progress = float(percent)
                job.stage = message
                store.persist(job)
            else:
                job.note(line)

        await process.wait()
    finally:
        canceller.cancel()

    if cancel.is_set():
        raise asyncio.CancelledError

    if process.returncode != 0:
        tail = "\n".join(job.log_tail[-12:])
        raise RuntimeError(
            f"Schritt fehlgeschlagen (Exit-Code {process.returncode}).\n{tail}"
        )


async def run_job(job: Job, store: JobStore, cancel: asyncio.Event) -> None:
    job.status = RUNNING
    job.started_at = time.time()
    job.progress = 1.0
    job.stage = "Wird gestartet"
    store.persist(job)

    scene_dir = job.scene_dir
    scene_dir.mkdir(parents=True, exist_ok=True)

    await _stream_process(
        [
            sys.executable, "-m", "kartograph.runner_inference",
            "--video", job.video_path,
            "--out-dir", str(scene_dir),
        ],
        job, store, cancel,
    )

    inference_meta = scene_dir / "inference.json"
    if inference_meta.exists():
        meta = json.loads(inference_meta.read_text(encoding="utf-8"))
        job.device = meta.get("device")

    await _stream_process(
        [
            sys.executable, "-m", "kartograph.export_scene",
            "--npz", str(scene_dir / "predictions.npz"),
            "--out-dir", str(scene_dir),
        ],
        job, store, cancel,
    )

    scene_file = scene_dir / "scene.json"
    if not scene_file.exists():
        raise RuntimeError("Export lief durch, hat aber keine scene.json hinterlassen.")

    job.scene = json.loads(scene_file.read_text(encoding="utf-8"))
    job.status = DONE
    job.progress = 100.0
    job.stage = f"Fertig — {job.scene['points']:,} Punkte".replace(",", ".")
    job.finished_at = time.time()
    store.persist(job)


async def worker(store: JobStore, cancels: dict[str, asyncio.Event]) -> None:
    """Endlosschleife: Job aus der Warteschlange holen, abarbeiten, wiederholen."""
    while True:
        job_id = await store.queue.get()
        job = store.get(job_id)

        if job is None or job.status != QUEUED:
            # Zwischenzeitlich abgebrochen oder gelöscht.
            store.queue.task_done()
            continue

        cancel = cancels.setdefault(job_id, asyncio.Event())

        try:
            await run_job(job, store, cancel)
        except asyncio.CancelledError:
            job.status = CANCELLED
            job.stage = "Abgebrochen"
            job.finished_at = time.time()
            store.persist(job)
        except Exception as exc:  # noqa: BLE001 - ein Job darf den Worker nie töten
            job.status = FAILED
            job.stage = "Fehlgeschlagen"
            job.error = str(exc)
            job.finished_at = time.time()
            store.persist(job)
        finally:
            cancels.pop(job_id, None)
            store.queue.task_done()


def cleanup_intermediates(scene_dir: Path, keep_npz: bool = True) -> None:
    """Die entpackten Frames wegräumen — die sind schnell mehrere hundert MB."""
    frames = scene_dir / "frames"
    if frames.exists():
        for frame in frames.glob("*.jpg"):
            frame.unlink(missing_ok=True)
        frames.rmdir()
    if not keep_npz:
        (scene_dir / "predictions.npz").unlink(missing_ok=True)
