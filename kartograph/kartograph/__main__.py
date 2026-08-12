"""Startpunkt: ``python -m kartograph``."""

from __future__ import annotations

import argparse
import os
import sys

from kartograph import config


def preflight() -> list[str]:
    """Vor dem Start prüfen, was fehlt — lieber jetzt als beim ersten Upload."""
    problems = []

    if not config.LINGBOT_REPO.exists():
        problems.append(
            f"LingBot-Map fehlt unter {config.LINGBOT_REPO}\n"
            "    → ./setup.sh ausführen, oder LINGBOT_REPO auf einen Klon zeigen lassen"
        )
    if not config.MODEL_PATH.exists():
        problems.append(
            f"Checkpoint fehlt unter {config.MODEL_PATH}\n"
            "    → ./setup.sh lädt ihn (4,6 GB), oder LINGBOT_MODEL setzen"
        )

    try:
        import torch  # noqa: F401
    except ImportError:
        problems.append("PyTorch ist nicht installiert → pip install -r requirements.txt")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(prog="kartograph", description=__doc__)
    parser.add_argument("--host", default=config.HOST)
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument(
        "--skip-checks", action="store_true",
        help="Trotz fehlendem Modell starten (die Oberfläche funktioniert dann schon)",
    )
    args = parser.parse_args()

    problems = preflight()
    if problems:
        print(f"\n  {config.APP_NAME} ist noch nicht startklar:\n")
        for problem in problems:
            print(f"  ✗ {problem}")
        print()
        if not args.skip_checks:
            print("  Mit --skip-checks trotzdem starten (Uploads schlagen dann fehl).\n")
            return 1

    import uvicorn

    # Den tatsächlich benutzten Port zurückschreiben, sonst nennt der QR-Code
    # weiter den Standardport und das Handy landet im Nichts.
    config.PORT = args.port
    os.environ["KARTOGRAPH_PORT"] = str(args.port)

    config.ensure_dirs()
    uvicorn.run("kartograph.server:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
