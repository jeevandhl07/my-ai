# My AI

My first small AI chatbot.

Current training accuracy: **90.79%**

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
