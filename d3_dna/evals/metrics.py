"""
D3-DNA evaluation metrics (dataset-agnostic).

All metric primitives in one file. Dataset-specific logic (oracle loading,
averaging, masking, protocol tiling) lives in each example directory, not here.

Oracle-based (require pre-computed (N, F) predictions):
    compute_fidelity_mse   -- paired MSE of oracle predictions
    compute_ks_statistic   -- mean per-feature two-sample KS statistic

Sequence-only (operate directly on (N, 4, L) one-hot arrays):
    compute_js_distance    -- Jensen-Shannon distance at a single k
    compute_js_spectrum    -- dict {k: JS distance} over a list of k values
    compute_auroc          -- DiscriminabilityCNN classifier AUROC (real=1, gen=0)
"""

import numpy as np
import scipy.spatial.distance
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
# Oracle-based metrics
# =============================================================================

def compute_fidelity_mse(pred_real, pred_gen):
    """Fidelity = (1/N) sum_i (f(x_real_i) - f(x_gen_i))^2, averaged over all feature dims."""
    return float(np.mean((np.asarray(pred_real) - np.asarray(pred_gen)) ** 2))


def compute_ks_statistic(pred_real, pred_gen, progress=False):
    """Mean over features j of sup_y |F_real_j(y) - F_gen_j(y)| (two-sample KS).

    pred_real, pred_gen: (N, F) arrays of oracle predictions.
    """
    pred_real = np.asarray(pred_real)
    pred_gen = np.asarray(pred_gen)
    n_features = pred_real.shape[1]
    stats = np.empty(n_features, dtype=np.float64)
    it = tqdm(range(n_features), desc="Per-feature KS") if progress else range(n_features)
    for j in it:
        stats[j] = scipy.stats.ks_2samp(pred_real[:, j], pred_gen[:, j]).statistic
    return float(stats.mean())


# =============================================================================
# k-mer spectrum shift: Jensen-Shannon distance
# =============================================================================

def _kmer_distribution(x, k):
    """Global k-mer frequency distribution pooled across all N sequences.

    x: (N, 4, L) one-hot (or soft; argmaxed). Returns length-4^k probability vector.
    """
    idx = np.asarray(x).argmax(axis=1).astype(np.int64)  # (N, L)
    L = idx.shape[1]
    if L < k:
        raise ValueError(f"Sequence length {L} shorter than k={k}")
    codes = np.zeros((idx.shape[0], L - k + 1), dtype=np.int64)
    for j in range(k):
        codes = codes * 4 + idx[:, j:L - k + 1 + j]
    counts = np.bincount(codes.ravel(), minlength=4 ** k).astype(np.float64)
    total = counts.sum()
    if total == 0:
        raise ValueError("Empty k-mer distribution")
    return counts / total


def compute_js_divergence(x_real, x_gen, k):
    """Jensen-Shannon divergence (base e) between k-mer distributions."""
    P = _kmer_distribution(x_real, k)
    Q = _kmer_distribution(x_gen, k)
    d = scipy.spatial.distance.jensenshannon(P, Q)
    return float(d ** 2)


def compute_js_spectrum(x_real, x_gen, ks):
    """Return {k: JS divergence} for each k in ks."""
    return {int(k): compute_js_divergence(x_real, x_gen, int(k)) for k in ks}


# =============================================================================
# Discriminability: binary CNN classifier, AUROC on a held-out split
# =============================================================================

class DiscriminabilityCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, 7, padding="same")
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(64, 96, 5, padding="same")
        self.bn2 = nn.BatchNorm1d(96)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(96, 128, 5, padding="same")
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        self.fc1 = nn.LazyLinear(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.LazyLinear(1)
        self.drop = nn.Dropout(0.2)
        self.drop_fc = nn.Dropout(0.5)

    def forward(self, x):
        x = self.drop(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = x.flatten(1)
        x = self.drop_fc(F.relu(self.bn4(self.fc1(x))))
        return self.fc2(x)


def compute_auroc(x_real, x_gen, epochs=50, batch_size=128, seed=42, device=None):
    """Train DiscriminabilityCNN on a 70/10/20 split (real=1, gen=0) with early stopping;
    return AUROC on the held-out 20% test set using the best-val checkpoint."""
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_real = np.asarray(x_real, dtype=np.float32)
    x_gen = np.asarray(x_gen, dtype=np.float32)
    X = np.concatenate([x_real, x_gen], axis=0)
    y = np.concatenate([np.ones(len(x_real)), np.zeros(len(x_gen))]).astype(np.float32)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    n_train = int(0.7 * len(X))
    n_val = int(0.1 * len(X))
    X_train, y_train = torch.tensor(X[:n_train]),               torch.tensor(y[:n_train])
    X_val,   y_val   = torch.tensor(X[n_train:n_train + n_val]), torch.tensor(y[n_train:n_train + n_val])
    X_test,  y_test  = torch.tensor(X[n_train + n_val:]),        torch.tensor(y[n_train + n_val:])

    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = DiscriminabilityCNN().to(device)
    with torch.no_grad():
        model(X_train[:2].to(device))  # init LazyLinear

    opt = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-6)
    crit = nn.BCEWithLogitsLoss()
    ds = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = None
    patience = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb).squeeze(), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device)).squeeze().cpu()
            vl = crit(val_logits, y_val).item()
        if vl < best_val - 0.001:
            best_val, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 10:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test.to(device)).squeeze().cpu().numpy()
    probs = 1 / (1 + np.exp(-test_logits))
    return float(roc_auc_score(y_test.numpy(), probs))
