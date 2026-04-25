"""Test helpers: thin wrapper over the public Zenodo utilities.

Preserves the legacy default-record behavior: tests fetch from the test
checkpoint record without having to thread the record through fixtures.
"""

from pathlib import Path

from d3_dna.utils.zenodo import fetch_zenodo as _fetch_zenodo

ZENODO_RECORD = "19488686"


def fetch_zenodo(filename: str, dest_dir, record: str = ZENODO_RECORD) -> Path:
    return _fetch_zenodo(filename, dest_dir, record=record)


__all__ = ["fetch_zenodo", "ZENODO_RECORD"]
