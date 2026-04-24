"""D3-DNA evaluation metrics (dataset-agnostic).

All metric primitives live in ``d3_dna.evals.metrics``. Dataset-specific
oracles and averaging/masking logic live in each ``examples/<dataset>/``.
"""

from d3_dna.evals.metrics import (
    compute_fidelity_mse,
    compute_ks_statistic,
    compute_js_divergence,
    compute_js_spectrum,
    compute_mean_js_divergence,
    compute_auroc,
    DiscriminabilityCNN,
)

__all__ = [
    "compute_fidelity_mse",
    "compute_ks_statistic",
    "compute_js_divergence",
    "compute_js_spectrum",
    "compute_mean_js_divergence",
    "compute_auroc",
    "DiscriminabilityCNN",
]
