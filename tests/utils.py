"""Test helpers: Zenodo download."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ZENODO_RECORD = "19488686"


def fetch_zenodo(filename: str, dest_dir: Path, record: str = ZENODO_RECORD) -> Path:
    """Download `filename` from Zenodo record into `dest_dir` if not already present.

    Returns the local path. Idempotent — skips the download if a non-empty file
    already exists at the destination.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    url = f"https://zenodo.org/records/{record}/files/{filename}?download=1"
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl not found on PATH; cannot download Zenodo checkpoint")

    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = [curl, "--fail", "--location", "--retry", "3", "-o", str(tmp), url]
    subprocess.run(cmd, check=True)
    tmp.replace(dest)
    return dest
