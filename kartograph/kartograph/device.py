"""Geräteauswahl für Apple Silicon, NVIDIA und CPU.

LingBot-Map wählt in ``demo.py`` fest zwischen CUDA und CPU — MPS kommt dort
nicht vor. Auf einem Mac liefe die Inferenz deshalb unbemerkt auf der CPU. Diese
Datei trifft die Wahl selbst und liefert dazu den passenden Datentyp.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceChoice:
    device: torch.device
    dtype: torch.dtype
    label: str
    note: str

    @property
    def is_mps(self) -> bool:
        return self.device.type == "mps"

    def as_dict(self) -> dict:
        return {
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "label": self.label,
            "note": self.note,
        }


def pick_device(prefer: str | None = None) -> DeviceChoice:
    """Bestes verfügbares Gerät wählen.

    ``prefer`` erzwingt ein Gerät ("mps", "cuda", "cpu") und ist vor allem zum
    Vergleichen da — etwa wenn ein MPS-Ergebnis seltsam aussieht und man
    gegenprüfen will, ob die CPU dasselbe liefert.
    """
    prefer = (prefer or os.environ.get("KARTOGRAPH_DEVICE") or "").strip().lower()

    if prefer == "cpu":
        return DeviceChoice(
            torch.device("cpu"), torch.float32, "CPU",
            "Erzwungen. Korrekt, aber deutlich langsamer als die GPU.",
        )

    if prefer in ("", "cuda") and torch.cuda.is_available():
        # bfloat16 erst ab Ampere (Compute Capability 8.0), davor float16.
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        return DeviceChoice(
            torch.device("cuda"), dtype, f"CUDA ({torch.cuda.get_device_name(0)})",
            "Voller Funktionsumfang inklusive FlashInfer.",
        )

    if prefer in ("", "mps") and torch.backends.mps.is_available():
        # float32 auf MPS: bfloat16 ist dort je nach macOS-Version noch
        # lückenhaft, und bei 36 GB Unified Memory ist der Speicher nicht der
        # Engpass. Wer experimentieren will, setzt KARTOGRAPH_MPS_DTYPE=bfloat16.
        dtype = torch.bfloat16 if os.environ.get("KARTOGRAPH_MPS_DTYPE") == "bfloat16" else torch.float32
        return DeviceChoice(
            torch.device("mps"), dtype, "Apple GPU (MPS)",
            "FlashInfer entfällt, es läuft PyTorch-SDPA.",
        )

    if prefer and prefer not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unbekanntes Gerät: {prefer!r} (erlaubt: mps, cuda, cpu)")

    return DeviceChoice(
        torch.device("cpu"), torch.float32, "CPU",
        "Keine GPU gefunden. Funktioniert, dauert aber lange.",
    )


def apply_mps_env() -> None:
    """Fehlende MPS-Kernel auf die CPU ausweichen lassen.

    Muss gesetzt sein, *bevor* torch die Backends initialisiert, sonst greift es
    nicht. Ohne den Schalter bricht die Inferenz bei der ersten nicht
    implementierten Operation mit NotImplementedError ab.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
