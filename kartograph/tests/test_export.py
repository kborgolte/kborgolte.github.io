"""Tests für die Rückprojektion und das Binärformat des Viewers.

Die Rekonstruktion selbst braucht GPU und einen 4,6-GB-Checkpoint und ist hier
nicht prüfbar. Die Mathematik dahinter schon: bei bekannter Kamera und bekannter
Tiefe steht vorher fest, wo ein Punkt landen muss. Genau das prüfen diese Tests
— dort säßen Vorzeichen- und Konventionsfehler, die man in einer Punktwolke
sonst erst bemerkt, wenn die Szene spiegelverkehrt ist.

Aufruf:  python tests/test_export.py
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kartograph.export_scene import (  # noqa: E402
    POINTS_MAGIC,
    choose_stride,
    export,
    unproject_frame,
)


def make_npz(path: Path, frames: int = 3, size: int = 64, depth_value: float = 5.0) -> Path:
    """Eine Szene bauen, deren korrektes Ergebnis von Hand nachrechenbar ist.

    Eine ebene Wand in konstanter Tiefe, die Kamera fährt in x-Richtung daran
    entlang.
    """
    focal = 100.0
    center = size / 2

    K = np.tile(
        np.array([[focal, 0, center], [0, focal, center], [0, 0, 1]], np.float32),
        (frames, 1, 1),
    )

    c2w = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    c2w[:, 0, 3] = np.arange(frames, dtype=np.float32)  # ein Meter pro Frame

    np.savez_compressed(
        path,
        images=np.full((frames, size, size, 3), 128, np.uint8),
        depth=np.full((frames, size, size), depth_value, np.float32),
        c2w=c2w,
        K=K,
        confidence=np.full((frames, size, size), 3.0, np.float32),
    )
    return path


def test_unprojection_geometry() -> None:
    """Ein Pixel in bekannter Tiefe muss an der berechneten Stelle landen."""
    size, focal, depth_value = 64, 100.0, 5.0
    K = np.array([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]], np.float32)

    xyz, _ = unproject_frame(
        depth=np.full((size, size), depth_value, np.float32),
        rgb=np.full((size, size, 3), 200, np.uint8),
        K=K,
        c2w=np.eye(4, dtype=np.float32),
        stride=1,
        max_depth=100.0,
        confidence=None,
        conf_threshold=0.0,
    )

    assert len(xyz) == size * size, f"{len(xyz)} Punkte statt {size * size}"

    # Ebene Wand, Kamera im Ursprung: alle Punkte müssen exakt in der Tiefe liegen.
    assert np.allclose(xyz[:, 2], depth_value), "Tiefe nicht erhalten"

    # Die Ecke bei Pixel (0,0) liegt bei (0 - 32)/100 * 5 = -1.6 in x und y.
    expected = -(size / 2) / focal * depth_value
    assert np.isclose(xyz[:, 0].min(), expected, atol=1e-4), f"x-Rand {xyz[:, 0].min()} statt {expected}"
    assert np.isclose(xyz[:, 1].min(), expected, atol=1e-4), f"y-Rand {xyz[:, 1].min()} statt {expected}"

    print("  ✓ Rückprojektion trifft die erwarteten Koordinaten")


def test_camera_pose_is_applied() -> None:
    """Verschiebt sich die Kamera, muss sich die Punktwolke mitverschieben."""
    size = 32
    K = np.array([[100, 0, 16], [0, 100, 16], [0, 0, 1]], np.float32)
    depth = np.full((size, size), 4.0, np.float32)
    rgb = np.zeros((size, size, 3), np.uint8)

    at_origin, _ = unproject_frame(depth, rgb, K, np.eye(4, dtype=np.float32), 1, 100.0, None, 0.0)

    shifted_pose = np.eye(4, dtype=np.float32)
    shifted_pose[0, 3] = 10.0
    shifted, _ = unproject_frame(depth, rgb, K, shifted_pose, 1, 100.0, None, 0.0)

    delta = shifted.mean(axis=0) - at_origin.mean(axis=0)
    assert np.allclose(delta, [10, 0, 0], atol=1e-4), f"Verschiebung {delta} statt [10,0,0]"

    print("  ✓ Kamerapose wird korrekt angewandt")


def test_confidence_filter() -> None:
    """Punkte unterhalb der Konfidenzschwelle müssen verschwinden."""
    size = 16
    K = np.array([[50, 0, 8], [0, 50, 8], [0, 0, 1]], np.float32)
    confidence = np.full((size, size), 3.0, np.float32)
    confidence[:8] = 0.5  # obere Hälfte unsicher

    xyz, _ = unproject_frame(
        np.full((size, size), 2.0, np.float32),
        np.zeros((size, size, 3), np.uint8),
        K, np.eye(4, dtype=np.float32), 1, 100.0, confidence, 1.5,
    )

    assert len(xyz) == size * 8, f"{len(xyz)} Punkte statt {size * 8} nach Filterung"
    print("  ✓ Konfidenzfilter greift")


def test_depth_range_filter() -> None:
    """Ungültige und zu weit entfernte Tiefen dürfen nicht in die Wolke."""
    size = 16
    K = np.array([[50, 0, 8], [0, 50, 8], [0, 0, 1]], np.float32)

    depth = np.full((size, size), 2.0, np.float32)
    depth[0] = 0.0        # Löcher, wie sie das Modell bei Himmel liefert
    depth[1] = 1e6        # Ausreißer
    depth[2] = np.nan     # NaN darf nicht durchrutschen

    xyz, _ = unproject_frame(
        depth, np.zeros((size, size, 3), np.uint8),
        K, np.eye(4, dtype=np.float32), 1, 100.0, None, 0.0,
    )

    assert len(xyz) == size * (size - 3), f"{len(xyz)} Punkte statt {size * (size - 3)}"
    assert np.isfinite(xyz).all(), "NaN oder Inf in der Punktwolke"
    print("  ✓ Tiefenfilter entfernt Löcher, Ausreißer und NaN")


def test_stride_respects_budget() -> None:
    assert choose_stride(10, 100, 100, budget=1_000_000) == 1, "kleine Szene braucht kein Ausdünnen"
    assert choose_stride(500, 518, 392, budget=100_000) > 4, "große Szene muss deutlich ausgedünnt werden"
    print("  ✓ Punktbudget steuert das Ausdünnen")


def test_export_writes_readable_files() -> None:
    """Ende zu Ende: NPZ rein, points.bin und GLB raus — im richtigen Format."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        npz = make_npz(tmp_path / "predictions.npz", frames=3, size=64)

        result = export(npz, tmp_path, max_points=100_000, conf_threshold=1.5)

        points_bin = tmp_path / "points.bin"
        assert points_bin.exists(), "points.bin fehlt"
        assert (tmp_path / "scene.json").exists(), "scene.json fehlt"

        raw = points_bin.read_bytes()
        assert raw[:4] == POINTS_MAGIC, "falsches Magic"

        version, count = struct.unpack("<II", raw[4:12])
        assert version == 1
        assert count == result["points"], f"Kopf meldet {count}, scene.json {result['points']}"

        # Genau so rechnet der Browser die Größe aus — passt das nicht, liest
        # der Viewer über das Pufferende hinaus.
        expected_bytes = 12 + count * 12 + count * 3
        assert len(raw) == expected_bytes, f"{len(raw)} Bytes statt {expected_bytes}"

        xyz = np.frombuffer(raw, "<f4", count=count * 3, offset=12).reshape(-1, 3)
        assert np.isfinite(xyz).all(), "NaN im Binärformat"
        assert np.allclose(xyz[:, 2], 5.0), "Tiefe im Export verändert"

        assert (tmp_path / "scene.glb").exists(), "GLB fehlt"
        assert result["glb_bytes"] > 0

        assert len(result["camera_path"]) == 3, "Kameraweg unvollständig"
        print(f"  ✓ Export erzeugt lesbare Dateien ({count:,} Punkte)".replace(",", "."))


def test_export_rejects_empty_scene() -> None:
    """Eine Szene ohne gültige Punkte muss klar scheitern, nicht leer durchlaufen."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        npz = make_npz(tmp_path / "empty.npz", frames=2, size=32, depth_value=0.0)

        try:
            export(npz, tmp_path, write_glb=False)
        except RuntimeError as exc:
            assert "Punkte" in str(exc)
            print("  ✓ Leere Szene wird mit verständlicher Meldung abgelehnt")
            return

    raise AssertionError("Leere Szene hätte einen Fehler auslösen müssen")


if __name__ == "__main__":
    print("\nExport-Tests\n")
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"\n{len(tests)} Tests bestanden.\n")
