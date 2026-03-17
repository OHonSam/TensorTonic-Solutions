import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    lengths = np.array([len(seq) for seq in seqs])
    if max_len is None:
        max_len = lengths.max()

    out = np.full(shape=(len(seqs), max_len), fill_value=pad_value)
    mask = np.arange(max_len) < lengths.reshape(-1, 1)
    flatten_seqs = np.concatenate([seq[: max_len] for seq in seqs])

    out[mask] = flatten_seqs

    return out