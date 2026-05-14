"""
Stub — train.py for the Zoonomia example.

Not wired: the Zoonomia dataloader yields X: (B, 350, 5) one-hot float and
y: (B, 1, 2), neither of which matches D3LightningModule's expected
(B, L) LongTensor / (B, signal_dim) FloatTensor contract. Calling D3Trainer
with this dataset would silently misroute, so we refuse instead.

See examples/zoonomia/README.md 'Not wired for training'.
"""

import argparse
import sys


def main(**_kwargs) -> None:
    print(
        "examples/zoonomia/train.py is not wired.\n"
        "The dataloader's X is (B, 350, 5) one-hot float and y is (B, 1, 2);\n"
        "D3LightningModule expects X: (B, L) LongTensor and y: (B, signal_dim).\n"
        "Before training, either collapse X to token ids via argmax(-1) or add\n"
        "a one-hot-consuming embedding head, and reshape y accordingly.\n"
        "See examples/zoonomia/README.md 'Not wired for training'.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=False)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--data-file", default=None)
    p.add_argument("--oracle-file", default=None)
    args = p.parse_args()
    main(**vars(args))
