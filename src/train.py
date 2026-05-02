"""Train the small character-level language model."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import SimpleLanguageModel


CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "ultrachat_lm.pt"


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible beginner-friendly experiments."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train the character model.")
    parser.add_argument("--data-path", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/ultrachat_lm.pt"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_existing_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load an existing checkpoint if one is available."""
    if not checkpoint_path.exists():
        return None
    return torch.load(checkpoint_path, map_location="cpu")


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    dataset: TextDataset,
    data_path: Path,
) -> None:
    """Ensure a checkpoint matches the current training configuration."""
    expected = {
        "vocab_size": dataset.tokenizer.vocab_size,
        "sequence_length": args.sequence_length,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "data_path": str(data_path),
    }

    for key, value in expected.items():
        stored_value = checkpoint.get(key)
        if stored_value is not None and stored_value != value:
            raise ValueError(
                f"Cannot resume: checkpoint {key} is {stored_value}, but current run expects {value}."
            )


def calculate_accuracy(
    model: SimpleLanguageModel,
    dataloader: DataLoader,
) -> float:
    """Measure next-character accuracy on the available dataset."""
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            logits = model(inputs)
            predicted_tokens = torch.argmax(logits, dim=-1)
            correct_predictions += (predicted_tokens == targets).sum().item()
            total_predictions += targets.numel()

    model.train()
    if total_predictions == 0:
        return 0.0
    return correct_predictions / total_predictions


def train(args: argparse.Namespace) -> None:
    """Train the model for a few epochs and save the checkpoint."""
    set_seed(args.seed)

    data_path = args.data_path.resolve()
    checkpoint_path = args.checkpoint_path.resolve()
    checkpoint_dir = checkpoint_path.parent
    existing_checkpoint = load_existing_checkpoint(checkpoint_path) if args.resume else None

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
    start_epoch = 0
    total_epochs_trained = 0

    if existing_checkpoint is not None:
        validate_resume_checkpoint(existing_checkpoint, args, dataset, data_path)
        model.load_state_dict(existing_checkpoint["model_state_dict"])
        optimizer_state = existing_checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        start_epoch = existing_checkpoint.get("last_epoch", 0)
        total_epochs_trained = existing_checkpoint.get("epochs_trained", start_epoch)
        print(f"Resuming from: {checkpoint_path}")

    best_loss = existing_checkpoint.get("best_loss", float("inf")) if existing_checkpoint else float("inf")

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
        current_epoch = start_epoch + epoch + 1

        if current_epoch % 50 == 0 or epoch == 0:
            print(f"Epoch {current_epoch} - Loss: {average_loss:.4f}")

        # Stop early once the model has mostly memorized this tiny training text.
        if average_loss < 0.015:
            print(f"Early stopping at epoch {current_epoch} with loss {average_loss:.4f}")
            break

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed_epochs = epoch + 1 if args.epochs > 0 else 0
    total_epochs_trained += completed_epochs
    train_accuracy = calculate_accuracy(model, dataloader)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stoi": dataset.tokenizer.stoi,
            "itos": dataset.tokenizer.itos,
            "vocab_size": dataset.tokenizer.vocab_size,
            "sequence_length": args.sequence_length,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "best_loss": best_loss,
            "data_path": str(data_path),
            "seed": args.seed,
            "epochs_trained": total_epochs_trained,
            "last_epoch": start_epoch + completed_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_accuracy": train_accuracy,
        },
        checkpoint_path,
    )
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    train(parse_args())
