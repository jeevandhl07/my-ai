"""A very small recurrent neural network for character prediction.

This is not a transformer. It uses:
- an embedding layer to turn token ids into vectors
- a GRU to learn order and context across the sequence
- a linear layer to predict the next character
"""

import torch
from torch import nn


class SimpleLanguageModel(nn.Module):
    """Small character-level model for learning the next token."""

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int = 8,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        """Initialize model layers."""
        super().__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for each position using recurrent sequence context."""
        embedded = self.embedding(x)
        hidden_states, _ = self.gru(embedded)
        logits = self.output(hidden_states)
        return logits
