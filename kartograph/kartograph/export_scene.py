#!/usr/bin/env python3
"""NPZ-Vorhersagen in eine Punktwolke umrechnen.

Das mitgelieferte ``demo_render/interactive_viewer/npz_to_glb.py`` behauptet im
Kopfkommentar, ohne CUDA-Abhängigkeiten auszukommen, ruft dann aber ``.cuda()``
auf den Tensoren auf — auf einem Mac bricht es damit ab. Die Rückprojektion ist
reine Kameramathematik, also rechnen wir sie hier selbst und geräteunabhängig.

Erzeugt zwei Dateien:
  scene.glb   für Blender, Preview.app und alles andere, was glTF liest
  points.bin  kompaktes Binärformat, das der Browser-Viewer direkt lädt

Aufruf:
    python -m kartograph.export_scene --npz scenes/abc/predictions.npz --out-dir scenes/abc
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from kartograph import config

# Kopf von points.bin: Magic, Version, Punktanzahl. 12 Bytes, durch 4 teilbar,
# damit der Float32Array im Browser ohne Kopie direkt darauf zeigen kann.
POINTS_MAGIC = b"KGPC"
POINTS_VERSION = 1


def progress(percent: float, message: str) -> None:
    print(f"KG_PROGRESS {percent:.1f} {message}", flush=True)


def log(message: str) -> None:
    print(f"[kartograph] {message}", flush=True)


def unproject_frame(
    depth: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    stride: int,
    max_depth: float,
    confidence: np.ndarray | None,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ein Tiefenbild in farbige Weltpunkte verwandeln."""
    d = depth[::stride, ::stride]
    color = rgb[::stride, ::stride]

    height, width = d.shape
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    us = us * stride
    vs = vs * stride

    valid = np.isfinite(d) & (d > 0) & (d < max_depth)
    if confidence is not None and conf_threshold > 0:
        valid &= confidence[::stride, ::stride] >= conf_threshold
    if not valid.any():
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    d = d[valid].astype(np.float32)
    us = us[valid].astype(np.float32)
    vs = vs[valid].astype(np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Lochkameramodell: Pixel zurück in Kamerakoordinaten, dann in die Welt.
    cam = np.stack([(us - cx) / fx * d, (vs - cy) / fy * d, d], axis=1)
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]

    return world.astype(np.float32), color[valid].astype(np.uint8)


def choose_stride(frames: int, height: int, width: int, budget: int) -> int:
    """Pixelraster so ausdünnen, dass das Punktbudget grob eingehalten wird.

    Lieber gleichmäßig über alle Frames ausdünnen als am Ende zufällig
    wegzuwerfen: das erhält die Struktur der Oberflächen sichtbar besser.
    """
    total = frames * height * width
    stride = 1
    # Faktor 1.5, weil Tiefen- und Konfidenzfilter ohnehin noch Punkte entfernen.
    while total / (stride**2) > budget * 1.5 and stride < 16:
        stride += 1
    return stride


def write_points_bin(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with path.open("wb") as handle:
        handle.write(POINTS_MAGIC)
        handle.write(struct.pack("<II", POINTS_VERSION, len(xyz)))
        handle.write(np.ascontiguousarray(xyz, dtype="<f4").tobytes())
        handle.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())


def export(
    npz_path: Path,
    out_dir: Path,
    max_points: int = config.VIEWER_MAX_POINTS,
    max_depth: float = 100.0,
    conf_threshold: float = 1.5,
    write_glb: bool = True,
) -> dict:
    data = np.load(npz_path)
    images = data["images"]
    depths = data["depth"]
    c2ws = data["c2w"]
    Ks = data["K"]
    confidence = data["confidence"] if "confidence" in data.files else None

    frames, height, width = depths.shape
    stride = choose_stride(frames, height, width, max_points)
    log(f"{frames} Frames à {width}x{height}, Pixelraster jeder {stride}. Punkt")

    chunks_xyz: list[np.ndarray] = []
    chunks_rgb: list[np.ndarray] = []

    for index in range(frames):
        xyz, rgb = unproject_frame(
            depths[index], images[index], Ks[index], c2ws[index],
            stride, max_depth,
            confidence[index] if confidence is not None else None,
            conf_threshold,
        )
        if len(xyz):
            chunks_xyz.append(xyz)
            chunks_rgb.append(rgb)
        if index % 10 == 0:
            progress(90 + 6 * index / max(frames, 1), f"Punktwolke wird gebaut ({index}/{frames})")

    if not chunks_xyz:
        raise RuntimeError(
            "Keine gültigen Punkte. Meist heißt das: zu wenig Kamerabewegung im "
            "Video, oder die Konfidenzschwelle ist zu hoch."
        )

    xyz = np.concatenate(chunks_xyz)
    rgb = np.concatenate(chunks_rgb)
    log(f"{len(xyz):,} Punkte rekonstruiert")

    if len(xyz) > max_points:
        keep = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        keep.sort()  # Sortiert bleibt der Speicherzugriff beim Schreiben linear.
        xyz, rgb = xyz[keep], rgb[keep]
        log(f"auf {len(xyz):,} Punkte reduziert (Viewer-Budget)")

    out_dir.mkdir(parents=True, exist_ok=True)
    progress(97, "Punktwolke wird geschrieben")
    write_points_bin(out_dir / "points.bin", xyz, rgb)

    # Ausreißer verzerren die Kamerastartposition, deshalb die Grenzen über
    # Perzentile statt über min/max bestimmen.
    low = np.percentile(xyz, 1, axis=0)
    high = np.percentile(xyz, 99, axis=0)
    center = ((low + high) / 2).tolist()
    extent = float(np.linalg.norm(high - low))

    result = {
        "points": int(len(xyz)),
        "stride": stride,
        "center": center,
        "extent": extent,
        "bounds": {"min": low.tolist(), "max": high.tolist()},
        "camera_path": c2ws[:, :3, 3].astype(float).tolist(),
        "points_bin_bytes": (out_dir / "points.bin").stat().st_size,
    }

    if write_glb:
        try:
            import trimesh

            progress(98, "GLB wird geschrieben")
            alpha = np.full((len(rgb), 1), 255, np.uint8)
            cloud = trimesh.PointCloud(vertices=xyz, colors=np.hstack([rgb, alpha]))
            glb_path = out_dir / "scene.glb"
            trimesh.Scene([cloud]).export(glb_path)
            result["glb_bytes"] = glb_path.stat().st_size
        except ImportError:
            log("trimesh fehlt — GLB wird übersprungen, points.bin reicht dem Viewer")
        except Exception as exc:  # pragma: no cover - Export ist Beiwerk
            log(f"GLB-Export fehlgeschlagen ({exc}) — points.bin ist trotzdem da")

    (out_dir / "scene.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    progress(99, f"{len(xyz):,} Punkte fertig")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="NPZ in eine Punktwolke umwandeln")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=config.VIEWER_MAX_POINTS)
    parser.add_argument("--max-depth", type=float, default=100.0)
    parser.add_argument("--conf-threshold", type=float, default=1.5)
    parser.add_argument("--no-glb", action="store_true")
    args = parser.parse_args()

    export(
        args.npz, args.out_dir,
        max_points=args.max_points,
        max_depth=args.max_depth,
        conf_threshold=args.conf_threshold,
        write_glb=not args.no_glb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
