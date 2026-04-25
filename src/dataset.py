"""Dataset helpers for next-character prediction.

The dataset turns text into small input/target pairs:
- input: a sequence of characters
- target: the same sequence shifted by one character
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

from tokenizer import CharTokenizer


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "input.txt"


def load_text(data_path: Path = DATA_PATH) -> str:
    """Read the raw training text file."""
    return data_path.read_text(encoding="utf-8").strip()


class TextDataset(Dataset):
    """Create training pairs for a character-level language model."""

    def __init__(self, sequence_length: int = 8, data_path: Path = DATA_PATH) -> None:
        """Load text, build tokenizer, and prepare training tokens."""
        self.data_path = data_path
        self.text = load_text(data_path)
        self.tokenizer = CharTokenizer(self.text)
        self.tokens = self.tokenizer.encode(self.text)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        """Return the number of available training examples."""
        return max(0, len(self.tokens) - self.sequence_length)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one input sequence and its shifted target sequence."""
        input_tokens = self.tokens[index : index + self.sequence_length]
        target_tokens = self.tokens[index + 1 : index + self.sequence_length + 1]

        x = torch.tensor(input_tokens, dtype=torch.long)
        y = torch.tensor(target_tokens, dtype=torch.long)
        return x, y
