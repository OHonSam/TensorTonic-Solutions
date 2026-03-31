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
        if len(word) > self.max_word_len:
            return [self.unk_token]
            
        subwords = []
        start = 0
        end = len(word)
        
        find_prefix = True
        prefix_unknown = True
        
        while start < end:
            substr = word[start:end]

            if start != 0:
                substr = "##" + substr
                
            if substr in self.vocab:
                if find_prefix:
                    prefix_unknown = False
                    find_prefix = False
                
                subwords.append(substr)
                start = end
                end = len(word)
                
            else:
                end -= 1        

        if (find_prefix and prefix_unknown) or (not find_prefix and start != len(word)):
            subwords.append(self.unk_token)
            
        return subwords
