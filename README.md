# my-ai

My first AI

Current training accuracy: 87%

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

Train with custom settings:

```bash
python src/train.py --epochs 800 --sequence-length 16
```

Resume training from the latest checkpoint:

```bash
python src/train.py --resume --epochs 400
```

## Generate

```bash
python src/generate.py
```

Generate with your own prompt:

```bash
python src/generate.py --prompt "hello" --max-new-tokens 40 --temperature 0.8
```

Generate with top-k sampling:

```bash
python src/generate.py --prompt "hello" --max-new-tokens 40 --temperature 0.8 --top-k 5
```

Show saved model info:

```bash
python src/generate.py --show-info
```
