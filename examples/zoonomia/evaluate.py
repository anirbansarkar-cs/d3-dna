"""
Stub — evaluate.py for the Zoonomia example.

Not wired: the dataloader's X/Y shapes diverge from D3's expected eval
contract (D3Evaluator wants (N, 4, L) one-hot real data and per-sample scalar
labels). See examples/zoonomia/README.md 'Not wired for training'.
"""

import sys


def main() -> None:
    print(
        "examples/zoonomia/evaluate.py is not wired.\n"
        "The dataloader yields X: (B, 350, 5) one-hot and y: (B, 1, 2), which\n"
        "does not match D3Evaluator's expected real-data shape (N, 4, L).\n"
        "See examples/zoonomia/README.md 'Not wired for training'.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
