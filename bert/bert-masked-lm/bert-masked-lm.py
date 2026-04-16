import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    mask_positions: np.ndarray,
    replace_probs: np.ndarray,
    random_tokens: np.ndarray,
    mask_token_id: int = 103
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: tuple of (np.ndarray masked_ids, np.ndarray labels) with masking applied
    """
    labels = ~mask_positions * -100 + token_ids * mask_positions
    print(f"labels: {labels}")
    print(f"replace_probs: {replace_probs}")
    print(f"random_tokens: {random_tokens}")
    
    masked_ids = token_ids * ~mask_positions + mask_positions * (
        (replace_probs < 0.8) * mask_token_id +
        ((replace_probs >= 0.8) & (replace_probs < 0.9)) * random_tokens +
        (replace_probs >= 0.9) * token_ids
    )
        
    
    return (masked_ids, labels)

class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token logits: hidden_states @ W + b
        """
        return hidden_states @ self.W + self.b