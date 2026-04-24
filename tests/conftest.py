"""Shared fixtures + marker config for the d3-dna test suite."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_PATH = Path(__file__).resolve().parent / "benchmarks.json"
DEFAULT_CHECKPOINT_CACHE = Path(os.environ.get(
    "D3_CHECKPOINT_CACHE", str(Path.home() / "scratch" / "d3-checkpoints")
))

# Shared-path defaults used by fixtures. Reads only — writes go to tmp_path or scratch.
PROMOTER_DATA = Path("/grid/koo/home/shared/d3/data/promoter/Promoter_data.npz")
K562_DATA = Path("/grid/koo/home/shared/d3/data/lentimpra/lenti_MPRA_K562_data.h5")
PROMOTER_ORACLE = Path("/grid/koo/home/shared/d3/oracle_weights/promoter/best.sei.model.pth.tar")


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="Also run tests marked @pytest.mark.slow",
    )


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--run-slow")
    cuda = torch.cuda.is_available()
    skip_slow = pytest.mark.skip(reason="slow test; pass --run-slow to enable")
    skip_gpu = pytest.mark.skip(reason="requires CUDA")
    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not cuda:
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Benchmark structure
# ---------------------------------------------------------------------------

@dataclass
class Benchmarks:
    raw: dict

    @property
    def tolerances(self) -> dict:
        return self.raw.get("tolerances", {})

    @property
    def sample_config(self) -> dict:
        return self.raw.get("sample_config", {})

    def expected(self, ckpt_name: str) -> Optional[dict]:
        """Return the stored metric dict for a checkpoint, or None if unset."""
        block = self.raw.get("checkpoints", {}).get(ckpt_name)
        if not block:
            return None
        if all(v is None for v in block.values()):
            return None
        return block

    def tolerance(self, metric: str) -> float:
        return float(self.tolerances.get(metric, 0.05))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def checkpoint_cache_dir() -> Path:
    DEFAULT_CHECKPOINT_CACHE.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CHECKPOINT_CACHE


@pytest.fixture(scope="session")
def fetch_checkpoint(checkpoint_cache_dir):
    """Factory fixture: `fetch_checkpoint("D3_Tran_Promoter.ckpt") -> Path`."""
    from tests.utils import fetch_zenodo

    def _fetch(filename: str) -> Path:
        return fetch_zenodo(filename, checkpoint_cache_dir)
    return _fetch


@pytest.fixture(scope="session")
def benchmarks() -> Benchmarks:
    with open(BENCHMARKS_PATH) as f:
        return Benchmarks(json.load(f))


@pytest.fixture(scope="session")
def example_sys_path(repo_root):
    """Yield a context that temporarily extends sys.path for an example dir.

    Usage in a test:
        def test_x(example_sys_path):
            with example_sys_path("promoter"):
                from data import PromoterDataset  # resolves inside examples/promoter
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx(example_name: str):
        path = str(repo_root / "examples" / example_name)
        inserted = path not in sys.path
        if inserted:
            sys.path.insert(0, path)
        try:
            yield path
        finally:
            if inserted:
                sys.path.remove(path)
    return _ctx


@pytest.fixture(scope="session")
def promoter_real_data():
    """Load the promoter test split one-hot (N, 4, 1024) from the shared NPZ."""
    import numpy as np
    if not PROMOTER_DATA.exists():
        pytest.skip(f"Promoter data NPZ not found at {PROMOTER_DATA}")
    return np.load(PROMOTER_DATA)["test"][:, :, :4].transpose(0, 2, 1).astype(np.float32)


@pytest.fixture(scope="session")
def k562_real_data():
    """Load the K562 test split one-hot (N, 4, 230) from the shared H5."""
    import h5py
    import numpy as np
    if not K562_DATA.exists():
        pytest.skip(f"K562 data H5 not found at {K562_DATA}")
    with h5py.File(K562_DATA, "r") as f:
        onehot = np.array(f["onehot_test"])  # (N, 230, 4)
    return np.transpose(onehot, (0, 2, 1)).astype(np.float32)


@pytest.fixture(scope="session")
def promoter_oracle(device, example_sys_path):
    """Load the SEI oracle for promoter evaluation."""
    if not PROMOTER_ORACLE.exists():
        pytest.skip(f"Promoter oracle not found at {PROMOTER_ORACLE}")
    with example_sys_path("promoter"):
        from oracle import load as load_sei
        return load_sei(str(PROMOTER_ORACLE), device)


@pytest.fixture(scope="session")
def k562_oracle(device, example_sys_path):
    """Load the LegNet oracle for K562 evaluation.

    Resolves the oracle checkpoint path by trying a short list of known
    shared-filesystem candidates; skips cleanly if none exist.
    """
    candidates = [
        Path("/grid/koo/home/shared/d3/oracle_weights/lentimpra/best_model_k562-epoch=15-val_pearson=0.706.ckpt"),
        Path("/grid/koo/home/shared/d3/oracle_weights/lentimpra/oracle_best_model_k562.ckpt"),
        Path("/grid/koo/home/shared/d3/oracle_weights/lentimpra/best_model_k562.ckpt"),
        Path("/grid/koo/home/shared/d3/oracle_weights/lentimpra/oracle_best_model.ckpt"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        pytest.skip("No K562 LegNet oracle checkpoint found under /grid/koo/home/shared/d3/oracle_weights/lentimpra/")
    with example_sys_path("k562"):
        from oracle import load as load_legnet
        return load_legnet(str(path), device)
