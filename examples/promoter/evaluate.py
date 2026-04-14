"""
Evaluate generated promoter sequences.

Computes 4 metrics:
  - MSE: Oracle prediction MSE between test and generated sequences
  - KS statistic: Kolmogorov-Smirnov test on oracle prediction distributions
  - JS distance: Jensen-Shannon divergence of k-mer frequency distributions
  - AUROC: Binary classifier discriminability (real vs synthetic)

Usage:
    python evaluate.py --samples generated_test/samples.npz
    python evaluate.py --samples generated_test/samples.npz --test "mse,js_distance"
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import scipy.spatial.distance
import scipy.special
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_promoter_data(data_path):
    """Load promoter NPZ and extract one-hot sequences as (N, 4, seq_len)."""
    data = np.load(data_path)
    x_test = data["test"][:, :, :4].transpose(0, 2, 1)
    x_train = data["train"][:, :, :4].transpose(0, 2, 1)
    return x_test, x_train


def load_samples(samples_path):
    """Load generated samples NPZ as (N, 4, seq_len)."""
    data = np.load(samples_path)
    arr = data["arr_0"]  # (N, seq_len, 4)
    return arr.transpose(0, 2, 1)


def pad_to_4096(x):
    """Pad (N, 4, L) sequences to (N, 4, 4096) with uniform 0.25 background."""
    L = x.shape[-1]
    if L >= 4096:
        return x[:, :, :4096]
    pad_left = (4096 - L) // 2
    pad_right = 4096 - L - pad_left
    left = np.full((x.shape[0], 4, pad_left), 0.25)
    right = np.full((x.shape[0], 4, pad_right), 0.25)
    return np.concatenate([left, x, right], axis=-1)


# ---------------------------------------------------------------------------
# SEI oracle loading
# ---------------------------------------------------------------------------

def load_sei_oracle(oracle_path):
    """Load SEI oracle model."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from d3_dna.sei import Sei, NonStrandSpecific

    sei = Sei(4096, 21907)
    oracle = NonStrandSpecific(sei)

    checkpoint = torch.load(oracle_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Strip 'module.' prefix
    pattern = re.compile(r"^module\.")
    state_dict = {pattern.sub("", k): v for k, v in state_dict.items()}

    oracle.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle = oracle.to(device).eval()
    print(f"SEI oracle loaded on {device}")
    return oracle, device


def oracle_predict(oracle, x_np, device, batch_size=8):
    """Run oracle predictions in batches. x_np is (N, 4, 4096) numpy array."""
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(x_np), batch_size), desc="Oracle predictions"):
            batch = torch.tensor(x_np[i : i + batch_size], dtype=torch.float32, device=device)
            pred = oracle(batch).cpu().numpy()
            predictions.append(pred)
    return np.concatenate(predictions, axis=0)


# ---------------------------------------------------------------------------
# Metric 1: MSE (conditional generation fidelity)
# ---------------------------------------------------------------------------

def compute_mse(pred_test, pred_syn):
    """MSE between oracle predictions on test vs synthetic sequences."""
    return float(np.mean((pred_test - pred_syn) ** 2))


# ---------------------------------------------------------------------------
# Metric 2: KS statistic (predictive distribution shift)
# ---------------------------------------------------------------------------

def compute_ks_statistic(pred_test, pred_syn):
    """Mean over features j of the 2-sample KS statistic
    sup_y |F_gen_j(y) - F_real_j(y)| computed across the N samples."""
    F = pred_test.shape[1]
    stats = np.empty(F, dtype=np.float64)
    for j in tqdm(range(F), desc="Per-feature KS"):
        stats[j] = scipy.stats.ks_2samp(pred_test[:, j], pred_syn[:, j]).statistic
    return float(stats.mean())


# ---------------------------------------------------------------------------
# Metric 3: JS distance (k-mer spectrum shift)
# ---------------------------------------------------------------------------

class KmerFeaturizer:
    """Compute k-mer occurrence counts for DNA sequences."""

    def __init__(self, k):
        self.k = k
        self.letters = ["A", "C", "G", "T"]
        self.multiply_by = 4 ** np.arange(k - 1, -1, -1)
        self.n = 4 ** k

    def kmer_index(self, kmer):
        digits = np.array([self.letters.index(c) for c in kmer])
        return int((digits * self.multiply_by).sum())

    def featurize(self, seq):
        n_kmers = len(seq) - self.k + 1
        counts = np.zeros(self.n)
        for i in range(n_kmers):
            counts[self.kmer_index(seq[i : i + self.k])] += 1
        return counts


def onehot_to_strings(x_np):
    """Convert (N, 4, L) one-hot numpy to list of DNA strings."""
    mapping = {0: "A", 1: "C", 2: "G", 3: "T"}
    seqs = []
    indices = x_np.argmax(axis=1)  # (N, L)
    for row in indices:
        seqs.append("".join(mapping[i] for i in row))
    return seqs


def compute_js_distance(x_test, x_syn, k=6):
    """Jensen-Shannon divergence between k-mer frequency distributions.

    x_test, x_syn are (N, 4, L) numpy arrays (unpadded preferred).
    """
    featurizer = KmerFeaturizer(k)

    def kmer_distribution(x):
        strings = onehot_to_strings(x)
        counts = np.zeros(featurizer.n)
        for s in tqdm(strings, desc=f"Computing {k}-mers"):
            counts += featurizer.featurize(s)
        return counts / counts.sum()

    dist_test = kmer_distribution(x_test)
    dist_syn = kmer_distribution(x_syn)
    return float(scipy.spatial.distance.jensenshannon(dist_test, dist_syn))


# ---------------------------------------------------------------------------
# Metric 4: AUROC (discriminability)
# ---------------------------------------------------------------------------

class DiscriminabilityCNN(nn.Module):
    """Simple CNN binary classifier for real vs synthetic sequences."""

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


def compute_auroc(x_test, x_syn, epochs=50, batch_size=128, seed=42):
    """Train a binary classifier and return AUROC.

    x_test, x_syn are (N, 4, L) numpy arrays.
    """
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prepare data: label real=1, synthetic=0
    X = np.concatenate([x_test, x_syn], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(len(x_test)), np.zeros(len(x_syn))]).astype(np.float32)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # 80/20 split
    n_train = int(0.8 * len(X))
    X_train, y_train = torch.tensor(X[:n_train]), torch.tensor(y[:n_train])
    X_val, y_val = torch.tensor(X[n_train:]), torch.tensor(y[n_train:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiscriminabilityCNN().to(device)

    # Warmup forward pass for LazyLinear
    with torch.no_grad():
        model(X_train[:2].to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb).squeeze(), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device)).squeeze().cpu()
            val_loss = criterion(val_logits, y_val).item()

        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    # Evaluate with best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_logits = model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze().cpu().numpy()
    probs = 1 / (1 + np.exp(-all_logits))
    auroc = roc_auc_score(y[idx.argsort()], probs[idx.argsort()])  # unshuffle
    # Actually simpler: just compute on the shuffled data
    auroc = roc_auc_score(y, probs)
    return float(auroc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = ["mse", "ks_statistic", "js_distance", "auroc"]
ORACLE_TESTS = {"mse", "ks_statistic"}

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated promoter sequences")
    parser.add_argument("--samples", required=True, help="Path to generated samples NPZ (arr_0 key)")
    parser.add_argument("--data", default="/grid/koo/home/shared/d3/data/promoter/Promoter_data.npz",
                        help="Path to promoter data NPZ")
    parser.add_argument("--model", default="/grid/koo/home/shared/d3/oracle_weights/promoter/best.sei.model.pth.tar",
                        help="Path to SEI oracle checkpoint")
    parser.add_argument("--test", default=None,
                        help="Comma-separated list of tests (default: all). Options: mse, ks_statistic, js_distance, auroc")
    parser.add_argument("--output-dir", default="eval_results", help="Output directory")
    parser.add_argument("--kmer-k", type=int, default=6, help="k-mer length for JS distance")
    args = parser.parse_args()

    tests = [t.strip() for t in args.test.split(",")] if args.test else ALL_TESTS

    # Load data (unpadded for k-mer analysis)
    print("Loading data...")
    x_test_raw, x_train_raw = load_promoter_data(args.data)
    x_syn_raw = load_samples(args.samples)
    print(f"  Test: {x_test_raw.shape}, Train: {x_train_raw.shape}, Synthetic: {x_syn_raw.shape}")

    # Oracle predictions (shared by MSE and KS)
    oracle_preds = {}
    needs_oracle = any(t in ORACLE_TESTS for t in tests)
    if needs_oracle:
        oracle, device = load_sei_oracle(args.model)
        x_test_padded = pad_to_4096(x_test_raw)
        x_syn_padded = pad_to_4096(x_syn_raw)
        print("Running oracle predictions on test sequences...")
        oracle_preds["test"] = oracle_predict(oracle, x_test_padded, device)
        print("Running oracle predictions on synthetic sequences...")
        oracle_preds["syn"] = oracle_predict(oracle, x_syn_padded, device)
        # Free GPU memory
        del oracle
        torch.cuda.empty_cache()

    results = {}

    for test_name in tests:
        print(f"\n=== {test_name} ===")
        if test_name == "mse":
            val = compute_mse(oracle_preds["test"], oracle_preds["syn"])
            results["mse"] = val
            print(f"  MSE: {val:.6f}")
        elif test_name == "ks_statistic":
            val = compute_ks_statistic(oracle_preds["test"], oracle_preds["syn"])
            results["ks_statistic"] = val
            print(f"  KS statistic: {val:.6f}")
        elif test_name == "js_distance":
            val = compute_js_distance(x_test_raw, x_syn_raw, k=args.kmer_k)
            results["js_distance"] = val
            print(f"  JS distance (k={args.kmer_k}): {val:.6f}")
        elif test_name == "auroc":
            val = compute_auroc(x_test_raw, x_syn_raw)
            results["auroc"] = val
            print(f"  AUROC: {val:.4f}")
        else:
            print(f"  Unknown test: {test_name}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
