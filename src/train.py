"""Train the small character-level language model."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import SimpleLanguageModel


CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "simple_lm.pt"


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible beginner-friendly experiments."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def train() -> None:
    """Train the model for a few epochs and save the checkpoint."""
    set_seed()

    sequence_length = 12
    dataset = TextDataset(sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = SimpleLanguageModel(
        vocab_size=dataset.tokenizer.vocab_size,
        sequence_length=sequence_length,
        embed_dim=32,
        hidden_dim=96,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 1500
    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0.0

        for inputs, targets in dataloader:
            optimizer.zero_grad()

            logits = model(inputs)
            loss = loss_fn(
                logits.view(-1, dataset.tokenizer.vocab_size),
                targets.view(-1),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / max(1, len(dataloader))
        best_loss = min(best_loss, average_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}")

        # Stop early once the model has mostly memorized this tiny training text.
        if average_loss < 0.015:
            print(f"Early stopping at epoch {epoch + 1} with loss {average_loss:.4f}")
            break

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stoi": dataset.tokenizer.stoi,
            "itos": dataset.tokenizer.itos,
            "vocab_size": dataset.tokenizer.vocab_size,
            "sequence_length": sequence_length,
            "embed_dim": 32,
            "hidden_dim": 96,
            "best_loss": best_loss,
        },
        CHECKPOINT_PATH,
    )
    print(f"Model saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
