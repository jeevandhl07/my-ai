"""Character-level tokenizer utilities.

This module keeps tokenization very simple:
- each unique character becomes one token id
- text is converted to integers with ``encode``
- integers are converted back to text with ``decode``
"""

from typing import Dict, List


class CharTokenizer:
    """A tiny character-level tokenizer for beginner experiments."""

    def __init__(self, text: str) -> None:
        """Build vocabulary tables from the training text."""
        unique_chars = sorted(set(text))
        self.stoi: Dict[str, int] = {char: index for index, char in enumerate(unique_chars)}
        self.itos: Dict[int, str] = {index: char for char, index in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        """Return the number of unique characters in the vocabulary."""
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        """Convert text into a list of token ids."""
        return [self.stoi[char] for char in text if char in self.stoi]

    def decode(self, tokens: List[int]) -> str:
        """Convert a list of token ids back into text."""
        return "".join(self.itos[token] for token in tokens)
