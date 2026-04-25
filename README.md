# my-ai

My first AI

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

## Generate

```bash
python src/generate.py
```

Generate with your own prompt:

```bash
python src/generate.py --prompt "hello" --max-new-tokens 40 --temperature 0.8
```
