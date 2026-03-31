from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
            
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords.
        """
        # Find prefix
        subwords = []
        suffix_start_idx = 0
        for i in range(len(word), 0, -1):
            prefix = word[:i]
            
            if prefix in self.vocab:
                subwords.append(prefix)
                suffix_start_idx = i
                break

        while suffix_start_idx < len(word):
            found_known_suffix = False
            for i in range(len(word), suffix_start_idx, -1):
                suffix = "##" + word[suffix_start_idx: i]
    
                if suffix in self.vocab:
                    subwords.append(suffix)
                    suffix_start_idx = i
                    found_known_suffix = True
                    break
                    
            if not found_known_suffix:
                subwords.append(self.unk_token)
                break
                
        return subwords
