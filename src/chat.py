"""Simple terminal chat loop for talking to the assistant continuously."""

import argparse
from pathlib import Path

from generate import CHECKPOINT_PATH, generate_reply


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the terminal chat loop."""
    parser = argparse.ArgumentParser(description="Chat with the assistant in a terminal loop.")
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Run a simple interactive chat session until the user exits."""
    args = parse_args()
    checkpoint_path = args.checkpoint_path.resolve()

    print("My AI chat started.")
    print("Type 'exit', 'quit', or 'bye' to stop.")

    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue

        if user_message.lower() in {"exit", "quit", "bye"}:
            print("AI: goodbye! talk to you later.")
            break

        assistant_reply = generate_reply(
            checkpoint_path=checkpoint_path,
            prompt=user_message,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"AI: {assistant_reply}")


if __name__ == "__main__":
    main()
