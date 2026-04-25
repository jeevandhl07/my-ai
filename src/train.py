"""Train the small character-level language model."""

import argparse
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train the Phase 2 character model.")
    parser.add_argument("--data-path", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/simple_lm.pt"))
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """Train the model for a few epochs and save the checkpoint."""
    set_seed(args.seed)

    data_path = args.data_path.resolve()
    checkpoint_path = args.checkpoint_path.resolve()
    checkpoint_dir = checkpoint_path.parent

    dataset = TextDataset(
        sequence_length=args.sequence_length,
        data_path=data_path,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SimpleLanguageModel(
        vocab_size=dataset.tokenizer.vocab_size,
        sequence_length=args.sequence_length,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")

    for epoch in range(args.epochs):
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
            print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {average_loss:.4f}")

        # Stop early once the model has mostly memorized this tiny training text.
        if average_loss < 0.015:
            print(f"Early stopping at epoch {epoch + 1} with loss {average_loss:.4f}")
            break

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stoi": dataset.tokenizer.stoi,
            "itos": dataset.tokenizer.itos,
            "vocab_size": dataset.tokenizer.vocab_size,
            "sequence_length": args.sequence_length,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "best_loss": best_loss,
            "data_path": str(data_path),
            "seed": args.seed,
        },
        checkpoint_path,
    )
    print(f"Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    train(parse_args())
