"""Generate text with the trained character-level model."""

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import torch

from model import SimpleLanguageModel


CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "personal_ai_checkpoint.pt"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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


def build_chat_seed(prompt: str) -> str:
    """Format the prompt to match the chat-style training data."""
    return f"user: {prompt} ai:"


def resolve_data_path(checkpoint: dict, checkpoint_path: Path) -> Path | None:
    """Choose the dataset file that best matches the selected checkpoint."""
    filename = checkpoint_path.name
    checkpoint_map = {
        "personal_ai_checkpoint.pt": DATA_DIR / "personal_input.txt",
        "ultrachat_lm.pt": DATA_DIR / "ultrachat_input.txt",
        "merged_ai_checkpoint.pt": DATA_DIR / "merged_input.txt",
    }

    mapped_path = checkpoint_map.get(filename)
    if mapped_path and mapped_path.exists():
        return mapped_path

    data_path = checkpoint.get("data_path")
    if not data_path:
        return None

    resolved = Path(data_path)
    if resolved.exists():
        return resolved

    return None


def load_conversation_pairs(
    checkpoint: dict,
    checkpoint_path: Path,
) -> list[tuple[str, str]]:
    """Load simple user/ai training pairs from the conversation dataset."""
    path = resolve_data_path(checkpoint, checkpoint_path)
    if path is None:
        return []

    pairs: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "user:" not in line or "ai:" not in line:
            continue

        user_part = line.split("user:", 1)[1].split("ai:", 1)[0].strip().lower()
        ai_part = line.split("ai:", 1)[1].split("<eos>", 1)[0].strip()
        pairs.append((user_part, ai_part))

    return pairs


def find_retrieved_reply(
    prompt: str,
    checkpoint: dict,
    checkpoint_path: Path,
) -> str | None:
    """Return the closest known reply for common prompts.

    This improves accuracy on the small set of conversation examples that the
    project currently knows well, while generation still handles the rest.
    """
    pairs = load_conversation_pairs(checkpoint, checkpoint_path)
    if not pairs:
        return None

    best_reply = None
    best_score = 0.0

    for known_prompt, known_reply in pairs:
        if prompt == known_prompt:
            return known_reply

        score = SequenceMatcher(a=prompt, b=known_prompt).ratio()
        if score > best_score:
            best_score = score
            best_reply = known_reply

    if best_score >= 0.86:
        return best_reply

    return None


def extract_reply(generated_text: str) -> str:
    """Return only the assistant reply portion of the generated text."""
    reply = generated_text

    if "ai:" in reply:
        reply = reply.split("ai:", 1)[1]

    if "<eos>" in reply:
        reply = reply.split("<eos>", 1)[0]

    if "user:" in reply:
        reply = reply.split("user:", 1)[0]

    return " ".join(reply.strip().split())


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


def generate_reply(
    checkpoint_path: Path,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.6,
    top_k: int = 3,
) -> str:
    """Return one assistant reply for a chat-style prompt."""
    checkpoint, _, _, _ = load_checkpoint(checkpoint_path)
    normalized_prompt = normalize_prompt(prompt)

    retrieved_reply = find_retrieved_reply(
        normalized_prompt,
        checkpoint,
        checkpoint_path,
    )
    if retrieved_reply is not None:
        return retrieved_reply

    seed_prompt = build_chat_seed(normalized_prompt)
    generated_text = generate_text(
        checkpoint_path=checkpoint_path,
        start_text=seed_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return extract_reply(generated_text)


if __name__ == "__main__":
    args = parse_args()
    checkpoint_path = args.checkpoint_path.resolve()
    checkpoint, _, _, _ = load_checkpoint(checkpoint_path)

    if args.show_info:
        print(format_checkpoint_info(checkpoint, checkpoint_path))
        raise SystemExit(0)

    prompt = resolve_prompt(args.prompt)
    if args.temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    if args.top_k < 0:
        raise ValueError("top_k must be 0 or greater.")

    output = generate_reply(
        checkpoint_path=checkpoint_path,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"Prompt: {normalize_prompt(prompt)}")
    print(f"Generated: {output}")
