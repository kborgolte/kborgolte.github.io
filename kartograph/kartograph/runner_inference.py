#!/usr/bin/env python3
"""Rekonstruktion eines Videos mit LingBot-Map, Ergebnis als NPZ.

Läuft absichtlich als eigener Prozess: der Webserver bleibt ansprechbar, der
Speicher wird am Ende garantiert freigegeben, und ein Abbruch ist ein simples
Kill. Fortschritt geht als ``KG_PROGRESS <prozent> <text>`` nach stdout,
``pipeline.py`` liest das mit.

Aufruf:
    python -m kartograph.runner_inference --video clip.mp4 --out-dir scenes/abc
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Muss vor dem torch-Import stehen, sonst greift der MPS-Fallback nicht.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402  (nach dem Env-Setup, siehe oben)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from kartograph import config  # noqa: E402
from kartograph.device import pick_device  # noqa: E402


def progress(percent: float, message: str) -> None:
    print(f"KG_PROGRESS {percent:.1f} {message}", flush=True)


def log(message: str) -> None:
    print(f"[kartograph] {message}", flush=True)


# ── Schritt 1: Frames aus dem Video ──────────────────────────────────────────

def extract_frames(video: Path, frames_dir: Path, target_fps: int, max_frames: int) -> list[Path]:
    """Frames gleichmäßig über das ganze Video verteilt herausschreiben.

    Gleichmäßig verteilt statt "die ersten N": bei einem zu langen Clip soll die
    Rekonstruktion den gesamten abgelaufenen Raum abdecken und nicht nach dem
    ersten Drittel enden.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Video lässt sich nicht öffnen: {video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total <= 0:
        # Manche Handy-Container melden keine Frame-Anzahl. Dann eben zählen.
        total = 0
        while capture.grab():
            total += 1
        capture.release()
        capture = cv2.VideoCapture(str(video))

    duration = total / source_fps if source_fps else 0.0
    wanted = int(round(duration * target_fps)) if duration else total
    wanted = max(2, min(wanted, max_frames, total))

    indices = np.unique(np.linspace(0, max(total - 1, 0), wanted).astype(int))
    log(f"Video: {total} Frames @ {source_fps:.1f} fps ({duration:.1f}s) → {len(indices)} Frames verwendet")

    wanted_set = set(indices.tolist())
    written: list[Path] = []
    position = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if position in wanted_set:
            path = frames_dir / f"frame_{len(written):05d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written.append(path)
            if len(written) % 20 == 0:
                progress(5 + 10 * len(written) / len(indices), f"Frames entpackt ({len(written)}/{len(indices)})")
        position += 1

    capture.release()

    if len(written) < 2:
        raise RuntimeError(
            f"Nur {len(written)} Frame(s) aus dem Video gewonnen — zu wenig für eine "
            "Rekonstruktion. Ist der Clip kürzer als eine Sekunde oder beschädigt?"
        )
    return written


# ── Schritt 2: Modell laden ──────────────────────────────────────────────────

def import_lingbot(repo: Path):
    """LingBot-Map importierbar machen und die Helfer aus demo.py holen."""
    if not repo.exists():
        raise RuntimeError(
            f"LingBot-Map nicht gefunden unter {repo}.\n"
            "Einmalig einrichten mit ./setup.sh oder LINGBOT_REPO auf den Klon zeigen lassen."
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    try:
        import demo  # type: ignore
        from lingbot_map.utils.load_fn import load_and_preprocess_images  # type: ignore
    except ImportError as exc:  # pragma: no cover - Umgebungsproblem
        raise RuntimeError(
            f"LingBot-Map lässt sich nicht importieren ({exc}). "
            "Fehlt 'pip install -e vendor/lingbot-map'?"
        ) from exc

    return demo, load_and_preprocess_images


def build_model_args(args, use_sdpa: bool) -> SimpleNamespace:
    """Das Argument-Objekt nachbauen, das demo.load_model erwartet."""
    return SimpleNamespace(
        mode=args.mode,
        model_path=str(args.model),
        image_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=True,
        max_frame_num=max(1024, args.max_frames + 64),
        kv_cache_sliding_window=64,
        num_scale_frames=args.num_scale_frames,
        use_sdpa=use_sdpa,
        camera_num_iterations=4,
    )


# ── Schritt 3: NPZ schreiben ─────────────────────────────────────────────────

def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").float().numpy()
    return np.asarray(value)


def save_predictions(predictions: dict, images: torch.Tensor, out_file: Path) -> dict:
    """Vorhersagen in das Schema schreiben, das der Exporter erwartet.

    ``postprocess`` liefert die Extrinsik bereits als c2w im 3x4-Format; für die
    Rückprojektion brauchen wir 4x4, also unten auffüllen.
    """
    depth = to_numpy(predictions["depth"])
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]  # [S, H, W, 1] → [S, H, W]

    extrinsic = to_numpy(predictions["extrinsic"])  # [S, 3, 4], c2w
    frames = extrinsic.shape[0]
    c2w = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    c2w[:, :3, :4] = extrinsic

    # postprocess entfernt "images" aus den Vorhersagen, *bevor* es die übrigen
    # Tensoren von der Batch-Dimension befreit — die Bilder kommen deshalb je
    # nach Pfad als [1, S, 3, H, W] oder [S, 3, H, W] zurück.
    rgb = to_numpy(images)  # 0..1
    if rgb.ndim == 5 and rgb.shape[0] == 1:
        rgb = rgb[0]
    rgb = np.clip(np.transpose(rgb, (0, 2, 3, 1)) * 255.0, 0, 255).astype(np.uint8)

    confidence = predictions.get("depth_conf")
    confidence = to_numpy(confidence) if confidence is not None else None

    payload = {
        "images": rgb,
        "depth": depth.astype(np.float32),
        "c2w": c2w.astype(np.float32),
        "K": to_numpy(predictions["intrinsic"]).astype(np.float32),
    }
    if confidence is not None:
        payload["confidence"] = confidence.astype(np.float32)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, **payload)

    return {
        "frames": int(frames),
        "height": int(depth.shape[1]),
        "width": int(depth.shape[2]),
        "has_confidence": confidence is not None,
        "npz_bytes": out_file.stat().st_size,
    }


# ── Ablauf ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="LingBot-Map-Rekonstruktion für Kartograph")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--lingbot-repo", type=Path, default=config.LINGBOT_REPO)
    parser.add_argument("--fps", type=int, default=config.DEFAULT_FPS)
    parser.add_argument("--max-frames", type=int, default=config.DEFAULT_MAX_FRAMES)
    parser.add_argument("--image-size", type=int, default=config.DEFAULT_IMAGE_SIZE)
    parser.add_argument("--patch-size", type=int, default=config.DEFAULT_PATCH_SIZE)
    parser.add_argument("--mode", choices=["streaming", "windowed"], default="streaming")
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--keyframe-interval", type=int, default=None)
    parser.add_argument("--device", default=None, help="mps, cuda oder cpu erzwingen")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Checkpoint fehlt: {args.model}\n"
            "Herunterladen mit ./setup.sh oder LINGBOT_MODEL setzen."
        )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    choice = pick_device(args.device)
    log(f"Gerät: {choice.label} ({choice.device}, {choice.dtype}) — {choice.note}")
    progress(2, f"Gerät gewählt: {choice.label}")

    progress(5, "Frames werden aus dem Video entpackt")
    frame_paths = extract_frames(args.video, out_dir / "frames", args.fps, args.max_frames)

    progress(16, "LingBot-Map wird geladen")
    demo, load_and_preprocess_images = import_lingbot(args.lingbot_repo)

    # FlashInfer gibt es nur für CUDA. Überall sonst muss SDPA einspringen —
    # ohne das bricht der Attention-Layer mit RuntimeError ab.
    use_sdpa = choice.device.type != "cuda"
    model = demo.load_model(build_model_args(args, use_sdpa), choice.device)

    if choice.dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        # Der Trunk darf in halber Genauigkeit laufen, die Köpfe bleiben fp32 —
        # so macht es demo.py auch, und es spart ein paar Gigabyte.
        model.aggregator = model.aggregator.to(dtype=choice.dtype)

    progress(22, f"{len(frame_paths)} Frames werden vorbereitet")
    images = load_and_preprocess_images(
        [str(p) for p in frame_paths],
        mode="crop",
        image_size=args.image_size,
        patch_size=args.patch_size,
    ).to(choice.device)

    progress(28, f"Rekonstruktion läuft ({len(frame_paths)} Frames, {choice.label})")
    log("Das ist der lange Teil — auf Apple-GPU grob 1-3 Frames pro Sekunde.")

    inference_started = time.time()
    autocast_enabled = choice.dtype != torch.float32
    with torch.no_grad(), torch.autocast(
        device_type=choice.device.type, dtype=choice.dtype, enabled=autocast_enabled
    ):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=torch.device("cpu"),
            )
        else:
            predictions = model.inference_windowed(
                images,
                window_size=64,
                overlap_size=16,
                overlap_keyframes=None,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=torch.device("cpu"),
            )
    inference_seconds = time.time() - inference_started
    log(f"Rekonstruktion fertig in {inference_seconds:.1f}s")

    progress(82, "Kamerapositionen werden umgerechnet")
    images_for_post = predictions.get("images", images)
    predictions, images_cpu = demo.postprocess(predictions, images_for_post)

    progress(88, "Ergebnis wird gespeichert")
    stats = save_predictions(predictions, images_cpu, out_dir / "predictions.npz")

    meta = {
        "device": choice.as_dict(),
        "mode": args.mode,
        "source_video": str(args.video),
        "frames_used": len(frame_paths),
        "sample_fps": args.fps,
        "inference_seconds": round(inference_seconds, 1),
        "total_seconds": round(time.time() - started, 1),
        "seconds_per_frame": round(inference_seconds / max(len(frame_paths), 1), 2),
        **stats,
    }
    (out_dir / "inference.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    progress(90, "Rekonstruktion abgeschlossen")
    log(f"Fertig in {meta['total_seconds']}s ({meta['seconds_per_frame']}s pro Frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
