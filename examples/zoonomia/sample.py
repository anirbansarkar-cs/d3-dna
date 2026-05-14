"""
Stub — sample.py for the Zoonomia example.

Not wired: see examples/zoonomia/README.md 'Not wired for training' and the
sibling train.py for the X/Y shape divergence that blocks training and
sampling.
"""

import argparse
import sys


def main(**_kwargs) -> None:
    print(
        "examples/zoonomia/sample.py is not wired.\n"
        "See examples/zoonomia/README.md 'Not wired for training'.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", default="config_transformer.yaml")
    p.add_argument("--output-dir", default="generated")
    args = p.parse_args()
    main(**vars(args))
