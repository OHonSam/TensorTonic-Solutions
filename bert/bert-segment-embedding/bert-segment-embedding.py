import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        
        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        
        # Segment embeddings (just 2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Compute BERT embeddings.
        """
        batch_size, seq_len = token_ids.shape
        cur_token_embs = self.token_embeddings[token_ids] # numpy array indexing
        cur_segment_embs = self.segment_embeddings[segment_ids]

        # Create absolute position indices: [0, 1, 2, ..., seq_len - 1]
        position_indices = np.arange(seq_len)
        cur_position_embs = self.position_embeddings[position_indices]

        return cur_token_embs + cur_position_embs + cur_segment_embs
            

        
