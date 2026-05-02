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
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=3)
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


def normalize_prompt(prompt: str) -> str:
    """Normalize prompt text to match the tiny training corpus better."""
    return prompt.strip().lower()


def load_training_text(checkpoint: dict) -> str:
    """Load the training text when the checkpoint knows where it came from."""
    data_path = checkpoint.get("data_path")
    if not data_path:
        return ""

    path = Path(data_path)
    if not path.exists():
        return ""

    return path.read_text(encoding="utf-8").strip().lower()


def choose_seed_prompt(prompt: str, checkpoint: dict, stoi: dict[str, int]) -> str:
    """Choose a more stable seed prompt for tiny-dataset generation.

    The current model learns from a very small training sentence, so prompts that
    are too short or not present in that sentence work better when we anchor them
    to a nearby known word from the training data.
    """
    training_text = load_training_text(checkpoint)
    if not training_text:
        filtered = "".join(char for char in prompt if char in stoi)
        return filtered or "my ai"

    filtered_prompt = "".join(char for char in prompt if char in stoi)
    if not filtered_prompt:
        return "my ai"

    greeting_expansions = {
        "hi": "hi there",
        "hey": "hey there",
        "hello": "hello there",
        "greetings": "greetings friend",
        "welcome": "welcome friend",
    }

    if filtered_prompt in greeting_expansions:
        return greeting_expansions[filtered_prompt]

    if filtered_prompt in training_text:
        return filtered_prompt

    words = training_text.split()
    best_word = ""
    best_score = -1

    for word in words:
        score = 0
        if filtered_prompt and word.startswith(filtered_prompt[0]):
            score += 3
        score += len(set(filtered_prompt) & set(word))
        if filtered_prompt in word or word in filtered_prompt:
            score += 2

        if score > best_score:
            best_score = score
            best_word = word

    if len(filtered_prompt) >= 2:
        return filtered_prompt

    return best_word or filtered_prompt


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
        f"last epoch: {checkpoint.get('last_epoch', 'n/a')}",
        f"batch size: {checkpoint.get('batch_size', 'n/a')}",
        f"learning rate: {checkpoint.get('learning_rate', 'n/a')}",
        f"training accuracy: {checkpoint.get('train_accuracy', 'n/a')}",
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
    checkpoint, model, stoi, itos = load_checkpoint(checkpoint_path)

    if args.show_info:
        print(format_checkpoint_info(checkpoint, checkpoint_path))
        raise SystemExit(0)

    prompt = normalize_prompt(resolve_prompt(args.prompt))
    if args.temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    if args.top_k < 0:
        raise ValueError("top_k must be 0 or greater.")

    seed_prompt = choose_seed_prompt(prompt, checkpoint, stoi)
    tokens = encode(seed_prompt, stoi)
    if not tokens:
        raise ValueError("Prompt does not contain any known characters from the training data.")

    for _ in range(args.max_new_tokens):
        context_window = tokens[-model.sequence_length :]
        if len(context_window) < model.sequence_length:
            pad_token = tokens[0]
            padding = [pad_token] * (model.sequence_length - len(context_window))
            context_window = padding + context_window

        x = torch.tensor([context_window], dtype=torch.long)

        with torch.no_grad():
            logits = model(x)

        next_token = sample_next_token(
            next_token_logits=logits[0, -1],
            temperature=args.temperature,
            top_k=args.top_k,
        )
        tokens.append(next_token)

    output = decode(tokens, itos)
    print(f"Prompt: {prompt}")
    if seed_prompt != prompt:
        print(f"Seed used: {seed_prompt}")
    print(f"Generated: {output}")
