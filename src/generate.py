"""Generate text with the trained character-level model."""

import argparse
from pathlib import Path

import torch

from model import SimpleLanguageModel


CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "simple_lm.pt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for text generation."""
    parser = argparse.ArgumentParser(description="Generate text with the character model.")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--show-info", action="store_true")
    return parser.parse_args()


def load_checkpoint(
    checkpoint_path: Path,
) -> tuple[dict, SimpleLanguageModel, dict[str, int], dict[int, str]]:
    """Load the saved model and vocabulary mappings."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train the model first with 'python src/train.py'."
        )

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

    return checkpoint, model, stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    """Convert prompt text into token ids."""
    return [stoi[char] for char in text if char in stoi]


def decode(tokens: list[int], itos: dict[int, str]) -> str:
    """Convert token ids back into readable text."""
    return "".join(itos[token] for token in tokens)


def sample_next_token(
    next_token_logits: torch.Tensor,
    temperature: float,
    top_k: int,
) -> int:
    """Sample the next token using temperature and optional top-k filtering."""
    scaled_logits = next_token_logits / temperature

    if top_k > 0:
        top_k = min(top_k, scaled_logits.size(0))
        values, indices = torch.topk(scaled_logits, k=top_k)
        probabilities = torch.softmax(values, dim=0)
        sampled_index = torch.multinomial(probabilities, num_samples=1).item()
        return indices[sampled_index].item()

    probabilities = torch.softmax(scaled_logits, dim=0)
    return torch.multinomial(probabilities, num_samples=1).item()


def resolve_prompt(prompt: str | None) -> str:
    """Use the provided prompt or ask the user for one in the terminal."""
    if prompt:
        return prompt

    typed_prompt = input("Enter a prompt: ").strip()
    if not typed_prompt:
        raise ValueError("Prompt cannot be empty.")
    return typed_prompt


def format_checkpoint_info(checkpoint: dict, checkpoint_path: Path) -> str:
    """Return a readable summary of the saved model settings."""
    lines = [
        f"checkpoint: {checkpoint_path}",
        f"vocab size: {checkpoint['vocab_size']}",
        f"sequence length: {checkpoint['sequence_length']}",
        f"embedding size: {checkpoint['embed_dim']}",
        f"hidden size: {checkpoint['hidden_dim']}",
        f"best loss: {checkpoint.get('best_loss', 'n/a')}",
        f"training data: {checkpoint.get('data_path', 'n/a')}",
        f"epochs trained: {checkpoint.get('epochs_trained', 'n/a')}",
        f"batch size: {checkpoint.get('batch_size', 'n/a')}",
        f"learning rate: {checkpoint.get('learning_rate', 'n/a')}",
    ]
    return "\n".join(lines)


def generate_text(
    checkpoint_path: Path,
    start_text: str = "hello",
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    top_k: int = 5,
) -> str:
    """Generate new text from a starting prompt.

    Temperature adds controlled randomness:
    - lower values make output safer and more repetitive
    - higher values make output more varied
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    if top_k < 0:
        raise ValueError("top_k must be 0 or greater.")

    _, model, stoi, itos = load_checkpoint(checkpoint_path)

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

        next_token_logits = logits[0, -1]
        next_token = sample_next_token(
            next_token_logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
        )
        tokens.append(next_token)

    return decode(tokens, itos)


if __name__ == "__main__":
    args = parse_args()
    checkpoint_path = args.checkpoint_path.resolve()
    checkpoint, _, _, _ = load_checkpoint(checkpoint_path)

    if args.show_info:
        print(format_checkpoint_info(checkpoint, checkpoint_path))
        raise SystemExit(0)

    prompt = resolve_prompt(args.prompt)
    output = generate_text(
        checkpoint_path=checkpoint_path,
        start_text=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {output}")
