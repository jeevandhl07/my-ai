"""High-confidence helpers for answering questions before neural generation.

The project model is intentionally tiny.  Retrieval gives exact, repeatable
answers for facts present in the training corpus, while the calculator handles
questions that should never be answered by probabilistic text generation.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path


WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
PAIR_RE = re.compile(
    r"user:\s*(.*?)\s+ai:\s*(.*?)(?=\s*<eos>|\s+user:|$)",
    flags=re.IGNORECASE,
)


def normalize_question(text: str) -> str:
    """Normalize casing, punctuation, and whitespace for stable matching."""
    return " ".join(WORD_RE.findall(text.casefold()))


def load_pairs(path: Path) -> list[tuple[str, str]]:
    """Parse every user/assistant pair, including multiple pairs on one line."""
    content = path.read_text(encoding="utf-8")
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in PAIR_RE.finditer(content):
        question = normalize_question(match.group(1))
        answer = " ".join(match.group(2).strip().split())
        pair = (question, answer)
        if question and answer and pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


@dataclass(frozen=True)
class Match:
    """A retrieved answer and its confidence score."""

    answer: str
    score: float
    question: str


class QAIndex:
    """Small dependency-free TF-IDF-like index for the local question corpus."""

    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs
        self.documents = [Counter(question.split()) for question, _ in pairs]
        document_frequency = Counter(
            token for document in self.documents for token in document
        )
        count = max(1, len(self.documents))
        self.idf = {
            token: math.log((count + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }

    def _cosine(self, query: Counter[str], document: Counter[str]) -> float:
        shared = query.keys() & document.keys()
        numerator = sum(
            query[token] * document[token] * self.idf.get(token, 1.0) ** 2
            for token in shared
        )
        query_norm = math.sqrt(
            sum((count * self.idf.get(token, 1.0)) ** 2 for token, count in query.items())
        )
        document_norm = math.sqrt(
            sum((count * self.idf.get(token, 1.0)) ** 2 for token, count in document.items())
        )
        if not query_norm or not document_norm:
            return 0.0
        return numerator / (query_norm * document_norm)

    def search(self, question: str) -> Match | None:
        normalized = normalize_question(question)
        if not normalized:
            return None

        # Exact matches are always safe, even when the corpus contains variants.
        for known_question, answer in self.pairs:
            if normalized == known_question:
                return Match(answer=answer, score=1.0, question=known_question)

        query = Counter(normalized.split())
        best: Match | None = None
        for (known_question, answer), document in zip(self.pairs, self.documents):
            cosine = self._cosine(query, document)
            character_similarity = SequenceMatcher(
                None, normalized, known_question
            ).ratio()
            coverage = len(query.keys() & document.keys()) / max(1, len(query))
            score = 0.55 * cosine + 0.30 * character_similarity + 0.15 * coverage
            if best is None or score > best.score:
                best = Match(answer=answer, score=score, question=known_question)

        # Short prompts require a stronger match because one shared word is noisy.
        # Longer prompts must share at least two meaningful terms, which accepts
        # natural paraphrases without treating a single generic word as evidence.
        threshold = 0.78 if len(query) <= 2 else 0.50
        if not best or best.score < threshold:
            return None
        if len(query) > 2 and len(query.keys() & set(best.question.split())) < 2:
            return None
        return best


@lru_cache(maxsize=8)
def build_index(path: Path) -> QAIndex:
    """Build and cache an index for a dataset path."""
    return QAIndex(load_pairs(path))


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 10 or abs(left) > 1_000_000):
            raise ValueError("Exponent is too large")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Unsupported expression")


def answer_math_question(question: str) -> str | None:
    """Safely answer a basic arithmetic question without using ``eval``."""
    text = question.casefold().strip().rstrip("?")
    prefixes = ("what is ", "calculate ", "compute ", "solve ")
    expression = next((text[len(prefix):] for prefix in prefixes if text.startswith(prefix)), text)
    expression = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
    if not expression or len(expression) > 100 or not re.fullmatch(r"[0-9+\-*/%.()\s]+", expression):
        return None
    try:
        result = _evaluate(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    if isinstance(result, float):
        if not math.isfinite(result):
            return None
        result = round(result, 10)
        if result.is_integer():
            result = int(result)
    return str(result)
