"""
D3-DNA post-hoc evaluation runner.

Thin, dataset-agnostic dispatcher over d3_dna.evals.metrics. The caller supplies
pre-loaded samples, pre-loaded real data, and a pre-loaded oracle. Any
dataset-specific logic (real-data file layout, oracle construction,
averaging/masking, DDSM tiling protocols) lives in the example directories.
"""

import json
import os
from typing import Optional

import numpy as np
import torch


class D3Evaluator:
    """Run the four D3 evaluation metrics on generated sequences.

    Usage::

        # caller owns oracle + real-data loading (typically inside an example's evaluate.py)
        from examples.promoter.oracle import load as load_sei
        oracle = load_sei(oracle_ckpt_path, device="cuda")
        real = _load_promoter_real(real_data_path)  # example-local helper

        ev = D3Evaluator()
        results = ev.evaluate(
            samples="samples.npz",
            real_data=real,
            oracle=oracle,
            tests=("mse", "ks", "js", "auroc"),
            kmer_ks=(6,),
            output_path="eval_results.json",
        )

    Supported tests: ``mse``, ``ks`` (oracle-based), ``js``, ``auroc``
    (sequence-only). ``oracle`` must expose a ``predict(x)`` method that maps
    (N, 4, L) one-hot to (N, F) oracle predictions; required for ``mse``/``ks``
    and ignored otherwise.
    """

    ALL_TESTS = ("mse", "ks", "js", "auroc")
    ORACLE_TESTS = {"mse", "ks"}

    def __init__(self, tests=None, device: Optional[str] = None):
        self.tests = tuple(tests) if tests else self.ALL_TESTS
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- I/O helpers ----

    @staticmethod
    def _normalize_onehot(x) -> np.ndarray:
        """Accept (N,4,L), (N,L,4), path to .npz, or torch tensor; return (N,4,L) float32."""
        if isinstance(x, str):
            data = np.load(x)
            key = "arr_0" if "arr_0" in data.files else data.files[0]
            x = data[key]
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim != 3:
            raise ValueError(f"Expected 3D sequence array, got shape {x.shape}")
        if x.shape[1] == 4:
            arr = x
        elif x.shape[2] == 4:
            arr = x.transpose(0, 2, 1)
        else:
            raise ValueError(f"Neither dim 1 nor dim 2 has size 4: {x.shape}")
        return arr.astype(np.float32)

    # ---- main entry point ----

    def evaluate(
        self,
        samples,
        real_data,
        oracle=None,
        tests=None,
        kmer_ks=(6,),
        output_path: Optional[str] = None,
    ) -> dict:
        from d3_dna.evals.metrics import (
            compute_fidelity_mse,
            compute_ks_statistic,
            compute_js_spectrum,
            compute_auroc,
        )

        tests = tuple(tests) if tests is not None else self.tests
        unknown = set(tests) - set(self.ALL_TESTS)
        if unknown:
            raise ValueError(f"Unknown tests: {sorted(unknown)}; valid: {self.ALL_TESTS}")

        x_real = self._normalize_onehot(real_data)
        x_gen = self._normalize_onehot(samples)

        if len(x_real) != len(x_gen):
            n = min(len(x_real), len(x_gen))
            print(f"[eval] truncating to paired N={n} (real={len(x_real)}, gen={len(x_gen)})")
            x_real, x_gen = x_real[:n], x_gen[:n]
        print(f"[eval] real={x_real.shape} gen={x_gen.shape}")

        results: dict = {}
        needs_oracle = any(t in self.ORACLE_TESTS for t in tests)
        pred_real = pred_gen = None
        if needs_oracle:
            if oracle is None:
                raise ValueError("oracle required for tests including mse/ks")
            print("[oracle] predicting on real...")
            pred_real = oracle.predict(x_real)
            print("[oracle] predicting on generated...")
            pred_gen = oracle.predict(x_gen)
            if self.device == "cuda":
                torch.cuda.empty_cache()

        for t in tests:
            print(f"=== {t} ===")
            if t == "mse":
                v = compute_fidelity_mse(pred_real, pred_gen)
                results["fidelity_mse"] = v
                print(f"  fidelity_mse: {v:.6f}")
            elif t == "ks":
                v = compute_ks_statistic(pred_real, pred_gen, progress=True)
                results["ks_statistic"] = v
                print(f"  ks_statistic: {v:.6f}")
            elif t == "js":
                spec = compute_js_spectrum(x_real, x_gen, kmer_ks)
                results["js_distance"] = {f"k{k}": v for k, v in spec.items()}
                for k, v in spec.items():
                    print(f"  js_distance k={k}: {v:.6f}")
            elif t == "auroc":
                v = compute_auroc(x_real, x_gen, device=self.device)
                results["auroc"] = v
                print(f"  auroc: {v:.6f}")

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"saved -> {output_path}")

        return results
