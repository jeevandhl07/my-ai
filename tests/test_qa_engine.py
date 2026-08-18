import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qa_engine import QAIndex, answer_math_question, load_pairs, normalize_question


class QAEngineTests(unittest.TestCase):
    def test_normalizes_punctuation_and_case(self):
        self.assertEqual(normalize_question("  What IS Python?! "), "what is python")

    def test_parses_multiple_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.txt"
            path.write_text(
                "user: hi ai: hello <eos> user: bye ai: goodbye <eos>",
                encoding="utf-8",
            )
            self.assertEqual(load_pairs(path), [("hi", "hello"), ("bye", "goodbye")])

    def test_exact_and_paraphrased_retrieval(self):
        index = QAIndex([
            ("what is python", "Python is a programming language."),
            ("how are you", "I am well."),
        ])
        self.assertEqual(index.search("What is Python?").score, 1.0)
        self.assertEqual(
            index.search("can you tell me what python is").answer,
            "Python is a programming language.",
        )

    def test_rejects_unrelated_question(self):
        index = QAIndex([("what is python", "A language.")])
        self.assertIsNone(index.search("who won the football match yesterday"))

    def test_answers_arithmetic_safely(self):
        self.assertEqual(answer_math_question("What is (12 + 8) * 3?"), "60")
        self.assertEqual(answer_math_question("calculate 7 / 2"), "3.5")
        self.assertIsNone(answer_math_question("what is __import__('os')"))
        self.assertIsNone(answer_math_question("hello"))


if __name__ == "__main__":
    unittest.main()
