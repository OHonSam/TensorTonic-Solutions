import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    new_seqs = []
    for seq in seqs:
        seq = np.array(seq)
        print("seq:", seq)
        
        if len(seq) <= max_len:
            num_pads = max_len - len(seq)
            required_pads = np.full(shape=num_pads, fill_value=pad_value)
            print("required_pads:", required_pads)
            new_seq = np.concatenate([seq, required_pads])
        else:
            print("seq:", seq[: max_len])
            new_seq = seq[: max_len]

        new_seqs.append(new_seq)
        
    return np.array(new_seqs)
            
            
        