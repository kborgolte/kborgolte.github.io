"""Zustand der Rekonstruktionsaufträge.

Bewusst schlicht gehalten: ein Dictionary im Speicher plus eine JSON-Datei pro
Job. Kein Datenbankserver für eine App, die auf genau einem Notebook läuft — und
nach einem Neustart sind die fertigen Szenen trotzdem noch da.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kartograph import config

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

TERMINAL = {DONE, FAILED, CANCELLED}


@dataclass
class Job:
    id: str
    name: str
    video_path: str
    status: str = QUEUED
    stage: str = "Wartet auf einen freien Platz"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    log_tail: list[str] = field(default_factory=list)
    scene: dict | None = None
    device: dict | None = None

    @property
    def scene_dir(self) -> Path:
        return config.SCENE_DIR / self.id

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration"] = round(
            (self.finished_at or time.time()) - (self.started_at or self.created_at), 1
        )
        data["has_glb"] = (self.scene_dir / "scene.glb").exists()
        return data

    def note(self, line: str) -> None:
        """Letzte Ausgabezeilen behalten — bei einem Fehler will man sie sehen."""
        self.log_tail.append(line)
        if len(self.log_tail) > 40:
            del self.log_tail[0]


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self.queue: asyncio.Queue[str] = asyncio.Queue()

    async def create(self, name: str, video_path: Path) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], name=name, video_path=str(video_path))
        job.scene_dir.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            self._jobs[job.id] = job
        self.persist(job)
        await self.queue.put(job.id)
        return job

    def add(self, job: Job) -> Job:
        """Einen fertig gebauten Job aufnehmen, ohne ihn einzureihen."""
        self._jobs[job.id] = job
        return job

    def remove(self, job_id: str) -> Job | None:
        return self._jobs.pop(job_id, None)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def persist(self, job: Job) -> None:
        try:
            (job.scene_dir / "job.json").write_text(
                json.dumps(job.to_dict(), indent=2), encoding="utf-8"
            )
        except OSError:
            # Ein fehlgeschlagener Schreibvorgang darf den Job nicht abbrechen.
            pass

    def load_from_disk(self) -> None:
        """Frühere Szenen wieder einlesen, damit sie nach einem Neustart da sind."""
        if not config.SCENE_DIR.exists():
            return
        for job_file in config.SCENE_DIR.glob("*/job.json"):
            try:
                raw = json.loads(job_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            known = {f for f in Job.__dataclass_fields__}
            job = Job(**{k: v for k, v in raw.items() if k in known})
            # Was beim Beenden noch lief, kann nicht fortgesetzt werden.
            if job.status in (RUNNING, QUEUED):
                job.status = FAILED
                job.error = "Beim Beenden der App unterbrochen"
            self._jobs[job.id] = job
