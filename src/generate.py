"""Generate text with the trained character-level model."""

import argparse
from pathlib import Path

import torch

from model import SimpleLanguageModel


CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "simple_lm.pt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for text generation."""
    parser = argparse.ArgumentParser(description="Generate text with the Phase 2 character model.")
    parser.add_argument("--prompt", type=str, default="hello")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    return parser.parse_args()


def load_checkpoint(
    checkpoint_path: Path,
) -> tuple[SimpleLanguageModel, dict[str, int], dict[int, str]]:
    """Load the saved model and vocabulary mappings."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    vocab_size = checkpoint["vocab_size"]
    sequence_length = checkpoint["sequence_length"]
    embed_dim = checkpoint["embed_dim"]
    hidden_dim = checkpoint["hidden_dim"]

    model = SimpleLanguageModel(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    """Convert prompt text into token ids."""
    return [stoi[char] for char in text if char in stoi]


def decode(tokens: list[int], itos: dict[int, str]) -> str:
    """Convert token ids back into readable text."""
    return "".join(itos[token] for token in tokens)


def generate_text(
    checkpoint_path: Path,
    start_text: str = "hello",
    max_new_tokens: int = 32,
    temperature: float = 0.8,
) -> str:
    """Generate new text from a starting prompt.

    Temperature adds controlled randomness:
    - lower values make output safer and more repetitive
    - higher values make output more varied
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")

    model, stoi, itos = load_checkpoint(checkpoint_path)

    tokens = encode(start_text, stoi)
    if not tokens:
        raise ValueError("Prompt does not contain any known characters from the training data.")

    for _ in range(max_new_tokens):
        context_window = tokens[-model.sequence_length :]
        if len(context_window) < model.sequence_length:
            pad_token = tokens[0]
            padding = [pad_token] * (model.sequence_length - len(context_window))
            context_window = padding + context_window

        x = torch.tensor([context_window], dtype=torch.long)

        with torch.no_grad():
            logits = model(x)

        next_token_logits = logits[0, -1] / temperature
        probabilities = torch.softmax(next_token_logits, dim=0)
        next_token = torch.multinomial(probabilities, num_samples=1).item()
        tokens.append(next_token)

    return decode(tokens, itos)


if __name__ == "__main__":
    args = parse_args()
    output = generate_text(
        checkpoint_path=args.checkpoint_path.resolve(),
        start_text=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"Prompt: {args.prompt}")
    print(f"Generated: {output}")
