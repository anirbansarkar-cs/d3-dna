"""Zenodo download + path resolution helpers used by examples to fetch
default datasets and oracle weights without hardcoded cluster paths."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union


def fetch_zenodo(
    filename: str,
    dest_dir: Union[str, Path],
    record: str,
    *,
    url: Optional[str] = None,
) -> Path:
    """Download `filename` from a Zenodo record into `dest_dir` if not present.

    Idempotent — skips the download when a non-empty file already exists at the
    destination. Pass `url` to download from an arbitrary HTTP(S) location
    instead (still cached at `dest_dir/filename`).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    if url is None:
        url = f"https://zenodo.org/records/{record}/files/{filename}?download=1"

    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl not found on PATH; cannot download from Zenodo")

    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = [curl, "--fail", "--location", "--retry", "3", "-o", str(tmp), url]
    subprocess.run(cmd, check=True)
    tmp.replace(dest)
    return dest


def resolve_path(
    spec: Optional[Union[str, Path]],
    cache_dir: Union[str, Path],
    filename: str,
    record: Optional[str] = None,
) -> Path:
    """Resolve a data-file path with override > local-cache > Zenodo-fetch layering.

    - If `spec` is a non-empty existing local path, return it (CLI override layer).
    - If `spec` is an http(s) URL, download to `cache_dir/filename` and return.
    - If `spec` is None and `cache_dir/filename` already exists, return that.
    - Else, if `record` is given, fetch `filename` from that Zenodo record into
      `cache_dir`. Without a record, raise — there is nothing to download.
    """
    cache_dir = Path(cache_dir)

    if spec is not None and str(spec):
        spec_str = str(spec)
        if spec_str.startswith(("http://", "https://")):
            return fetch_zenodo(filename, cache_dir, record="", url=spec_str)
        candidate = Path(spec_str)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Override path does not exist: {candidate}")

    cached = cache_dir / filename
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    if not record:
        raise FileNotFoundError(
            f"{cached} not found and no Zenodo record provided to download {filename}"
        )
    return fetch_zenodo(filename, cache_dir, record=record)
