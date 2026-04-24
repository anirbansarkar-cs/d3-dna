"""Regression test: sample + evaluate against pinned Zenodo checkpoints.

Each parametrized case:
    1. Downloads the checkpoint from Zenodo (cached after first run).
    2. Extracts the training config embedded in the checkpoint.
    3. Loads the model + sampler, generates N conditional samples.
    4. Scores those samples via D3Evaluator against the real test split
       (MSE, KS, JS, AUROC).
    5. Compares to tests/benchmarks.json within per-metric tolerance.
       If benchmarks are null (not yet captured), xfails with the
       observed values in the message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


@dataclass
class Case:
    name: str             # short id for parametrize
    ckpt_name: str        # Zenodo filename (without .ckpt)
    dataset: str          # "promoter" | "k562"
    steps_key: str        # key into benchmarks.sample_config


CASES = [
    Case("promoter-tran", "D3_Tran_Promoter",  "promoter", "steps_promoter"),
    Case("promoter-conv", "D3_Conv_Promoter",  "promoter", "steps_promoter"),
    Case("k562-tran",     "D3_Tran_MPRA_K562", "k562",     "steps_k562"),
    Case("k562-conv",     "D3_Conv_MPRA_K562", "k562",     "steps_k562"),
]


def _get_labels(dataset: str, n: int, repo_root, example_sys_path):
    """Pull the first `n` test-set labels for conditional generation."""
    if dataset == "promoter":
        with example_sys_path("promoter"):
            from data import PromoterDataset
            promoter_npz = "/grid/koo/home/shared/d3/data/promoter/Promoter_data.npz"
            test_ds = PromoterDataset(promoter_npz, split="test")
            return test_ds.y[:n]
    elif dataset == "k562":
        with example_sys_path("k562"):
            from data import K562Dataset
            k562_h5 = "/grid/koo/home/shared/d3/data/lentimpra/lenti_MPRA_K562_data.h5"
            test_ds = K562Dataset(k562_h5, split="test")
            return test_ds.y[:n]
    else:
        raise ValueError(f"unknown dataset: {dataset}")


def _build_model(cfg):
    from d3_dna.models import TransformerModel, ConvolutionalModel
    arch = cfg.model.architecture
    if arch == "transformer":
        return TransformerModel(cfg)
    elif arch == "convolutional":
        return ConvolutionalModel(cfg)
    else:
        raise ValueError(f"unknown architecture '{arch}'")


def _format_observed(results: dict) -> str:
    return ", ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in results.items())


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_sample_eval_benchmark(
    case: Case,
    device: str,
    repo_root,
    example_sys_path,
    fetch_checkpoint,
    benchmarks,
    promoter_real_data,
    k562_real_data,
    promoter_oracle,
    k562_oracle,
):
    # 1. Checkpoint
    ckpt_path = fetch_checkpoint(f"{case.ckpt_name}.ckpt")

    # 2. Config from checkpoint (authoritative)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("hyper_parameters", {}).get("cfg")
    assert raw_cfg is not None, f"Checkpoint has no embedded cfg: {ckpt_path}"
    cfg = OmegaConf.create(dict(raw_cfg))

    # 3. Model + sampler
    from d3_dna import D3Sampler
    model = _build_model(cfg)
    sampler = D3Sampler(cfg)
    sampler.load(checkpoint=str(ckpt_path), model=model, device=device)

    # 4. Labels + seeded sampling
    sconf = benchmarks.sample_config
    n = int(sconf.get("num_samples", 512))
    steps = int(sconf.get(case.steps_key, 128 if case.dataset == "promoter" else 20))
    batch_size = int(sconf.get("batch_size", 64))
    seed = int(sconf.get("seed", 0))

    labels = _get_labels(case.dataset, n, repo_root, example_sys_path)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    seqs = sampler.generate_batched(
        num_samples=n, labels=labels, batch_size=batch_size, steps=steps
    )

    # Minimal "sampling completes" checks — shape + value sanity.
    seq_len = cfg.dataset.sequence_length
    assert seqs.shape == (n, seq_len), f"Unexpected sample shape {seqs.shape}"
    assert seqs.min().item() >= 0 and seqs.max().item() < 4, (
        f"Tokens out of DNA vocab range [0,3]: min={seqs.min().item()}, max={seqs.max().item()}"
    )

    # 5. Evaluate via D3Evaluator
    from d3_dna import D3Evaluator
    onehot = F.one_hot(seqs.long().cpu(), num_classes=4).permute(0, 2, 1).numpy().astype(np.float32)
    real = (promoter_real_data if case.dataset == "promoter" else k562_real_data)[:n]
    oracle = promoter_oracle if case.dataset == "promoter" else k562_oracle

    ev = D3Evaluator(device=device)
    results = ev.evaluate(
        samples=onehot,
        real_data=real,
        oracle=oracle,
        tests=("mse", "ks", "js", "auroc"),
        kmer_ks=tuple(sconf.get("kmer_ks", [6])),
    )

    # 6. Compare to benchmark
    expected = benchmarks.expected(case.ckpt_name)
    if expected is None:
        pytest.xfail(f"no benchmark yet for {case.ckpt_name}; observed: {_format_observed(results)}")

    mismatches = []
    for metric, obs in results.items():
        exp = expected.get(metric)
        if exp is None:
            continue
        tol = benchmarks.tolerance(metric)
        if not math.isfinite(obs):
            mismatches.append(f"{metric}: observed={obs} (not finite)")
        elif abs(obs - exp) > tol:
            mismatches.append(f"{metric}: |{obs:.6f} - {exp:.6f}| > {tol}")

    assert not mismatches, (
        f"{case.ckpt_name} regression:\n"
        + "\n".join("  " + m for m in mismatches)
        + f"\n  observed full: {_format_observed(results)}"
    )
