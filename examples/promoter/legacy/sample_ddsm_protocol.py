"""5-per-TSS sampling on the 40k chr8/9 test split for D3 or DDSM checkpoints.

Runs from examples/promoter/legacy/. The vendored DDSM utilities live in the
sibling `ddsm_ref/` package. The DDSM pre-sampled noise schedule (a 3.6 GB
`.pth` file) is no longer bundled with the repo — pass its absolute path via
`--ddsm-schedule`.

Usage (D3):
    python sample_ddsm_protocol.py --model d3 \
        --checkpoint /grid/koo/home/shared/d3/trained_weights/promoter/model-epoch=79-val_loss=1102.0673.ckpt \
        --steps 100 --output-dir generated/d3_epoch79_5perTSS \
        --config ../config_transformer.yaml

Usage (DDSM):
    python sample_ddsm_protocol.py --model ddsm \
        --checkpoint /grid/koo/home/duran/ddsm_checkpoint.pth \
        --steps 100 --output-dir generated/ddsm_5perTSS_100 \
        --ddsm-schedule /grid/koo/home/shared/d3/ddsm_artifacts/steps400.cat4.speed_balance.time4.0.samples100000.pth
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_DATA = "/grid/koo/home/shared/d3/data/promoter/Promoter_data_40k.npz"
N_PER_TSS = 5


def load_cage_labels_5x(data_path=DEFAULT_DATA):
    d = np.load(data_path)
    test = d["test"]                                  # (N, 1024, 6)
    cage = test[:, :, 4:5].astype(np.float32)         # (N, 1024, 1)
    cage5 = np.repeat(cage, N_PER_TSS, axis=0)        # (5N, 1024, 1)
    tss_idx = np.repeat(np.arange(len(cage)), N_PER_TSS)
    return cage5, tss_idx


def sample_d3(checkpoint, labels, steps, batch_size, config_path):
    from omegaconf import OmegaConf
    from d3_dna import D3Sampler
    from d3_dna.models import TransformerModel

    cfg = OmegaConf.load(config_path)
    model = TransformerModel(cfg)
    sampler = D3Sampler(cfg)
    sampler.load(checkpoint=checkpoint, model=model, device="cuda")

    labels_t = torch.from_numpy(labels)               # (14525, 1024, 1)
    seqs = sampler.generate_batched(
        num_samples=len(labels_t),
        labels=labels_t,
        batch_size=batch_size,
        steps=steps,
    )                                                  # (14525, 1024) int tokens
    onehot = F.one_hot(seqs.long(), num_classes=4).numpy().astype(np.float32)
    return onehot


def sample_ddsm(checkpoint, labels, steps, batch_size, schedule_path):
    # Import the vendored DDSM utilities from the sibling dir
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from ddsm_ref import ScoreNet, Euler_Maruyama_sampler

    v_one, v_zero, v_one_lg, v_zero_lg, timepoints = torch.load(
        schedule_path, map_location="cpu", weights_only=False
    )
    max_time = float(timepoints[-1].item())
    min_time = float(timepoints[0].item())

    # time_dependent_weights is overwritten by checkpoint load; placeholder length must match (400)
    score_model = torch.nn.DataParallel(
        ScoreNet(time_dependent_weights=torch.zeros(400))
    ).cuda()
    score_model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True
    )
    score_model.eval()

    alpha = torch.FloatTensor([1.0, 1.0, 1.0]).cuda()
    beta = torch.FloatTensor([3.0, 2.0, 1.0]).cuda()

    labels_t = torch.from_numpy(labels)               # (14525, 1024, 1) float32
    all_onehot = []
    for i in range(0, len(labels_t), batch_size):
        batch = labels_t[i : i + batch_size].cuda()
        sb = Euler_Maruyama_sampler(
            score_model,
            (1024, 4),
            batch_size=batch.shape[0],
            max_time=max_time,
            min_time=min_time,
            time_dilation=1,
            num_steps=steps,
            eps=1e-5,
            alpha=alpha,
            beta=beta,
            speed_balanced=True,
            device="cuda",
            concat_input=batch,
        )                                              # (B, 1024, 4) simplex
        tokens = sb.argmax(dim=-1).cpu()
        onehot_b = F.one_hot(tokens, num_classes=4).numpy().astype(np.float32)
        all_onehot.append(onehot_b)
        print(f"  batch {i // batch_size + 1}: {onehot_b.shape}")
    return np.concatenate(all_onehot, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["d3", "ddsm"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--config", default="../config_transformer.yaml",
                   help="(d3 only) OmegaConf YAML for TransformerModel")
    p.add_argument("--ddsm-schedule", default=None,
                   help="(ddsm only) absolute path to pre-sampled noise schedule "
                        "(.pth file, e.g. steps400.cat4.speed_balance.time4.0.samples100000.pth)")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    if args.model == "ddsm" and not args.ddsm_schedule:
        p.error("--ddsm-schedule is required when --model=ddsm")

    os.makedirs(args.output_dir, exist_ok=True)
    labels, tss_idx = load_cage_labels_5x(args.data)
    print(f"labels={labels.shape} unique_tss={len(set(tss_idx))} total_samples={len(labels)}")

    if args.model == "d3":
        onehot = sample_d3(args.checkpoint, labels, args.steps,
                           args.batch_size, args.config)
    else:
        onehot = sample_ddsm(args.checkpoint, labels, args.steps,
                             args.batch_size, args.ddsm_schedule)

    out_path = os.path.join(args.output_dir, "sample_0.npz")
    np.savez(out_path, first_sample=onehot, tss_index=tss_idx)
    print(f"Saved {onehot.shape} to {out_path}")


if __name__ == "__main__":
    main()
