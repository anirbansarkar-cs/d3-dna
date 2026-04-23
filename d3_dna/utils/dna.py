"""DNA sequence helpers shared across the package."""


def sequences_to_strings(sequences, token_to_char=None):
    """Convert an integer token tensor (N, L) to a list of ACGT strings."""
    if token_to_char is None:
        token_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    out = []
    for seq in sequences:
        s = ''.join([token_to_char.get(int(t), 'N') for t in seq])
        out.append(s)
    return out
