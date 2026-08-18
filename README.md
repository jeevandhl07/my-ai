# My AI

My first small AI chatbot.

The assistant combines a tiny character-level GRU with high-confidence local
question retrieval and a safe arithmetic engine. Retrieval makes known and
similarly worded questions deterministic; low-confidence matches are rejected
instead of returning an unrelated memorized answer.

> The previous **90.79%** number measured next-character accuracy on the same
> text used for training. It was not question-answer accuracy. Use a held-out
> evaluation set before describing the model's real-world accuracy.

## Run This Project

Install packages:

```bash
python3 -m pip install -r requirements.txt
```

Train the AI:

```bash
python3 src/train.py --data-path data/merged_input.txt --checkpoint-path checkpoints/merged_ai_checkpoint.pt --epochs 300 --sequence-length 64
```

Chat with the AI:

```bash
python3 src/chat.py --checkpoint-path checkpoints/merged_ai_checkpoint.pt
```

Generate one reply:

```bash
python3 src/generate.py --checkpoint-path checkpoints/merged_ai_checkpoint.pt --prompt "hello"
```

Show saved model info:

```bash
python3 src/generate.py --checkpoint-path checkpoints/merged_ai_checkpoint.pt --show-info
```

Run the accuracy and safety tests:

```bash
python3 -m unittest discover -s tests -v
```

## How to teach it more answers

Add reviewed examples to `data/merged_input.txt` using this format, then train a
new checkpoint:

```text
user: what is react native ai: react native is a framework for building native mobile apps with react. <eos>
```

Prefer several natural phrasings for important questions. Keep factual answers
short, correct, and consistent; conflicting examples reduce answer quality.
