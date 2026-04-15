from __future__ import annotations

import unittest

from equant.evals.qa_eval import clean_prediction_text, stop_strings_for_dataset


class QAEvalFormattingTests(unittest.TestCase):
    def test_coqa_predictions_are_trimmed_to_first_answer_line(self) -> None:
        prediction = "Answer: blue\nQuestion: what else?\nAnswer: red"
        self.assertEqual(clean_prediction_text("coqa", prediction), "blue")

    def test_truthfulqa_predictions_drop_assistant_prefix(self) -> None:
        prediction = "assistant: Water boils at 100 C.\nExtra text"
        self.assertEqual(clean_prediction_text("truthfulqa", prediction), "Water boils at 100 C.")

    def test_gsm8k_predictions_keep_multi_line_reasoning(self) -> None:
        prediction = "Let's think.\nThe answer is 42."
        self.assertEqual(clean_prediction_text("gsm8k", prediction), prediction)

    def test_single_line_datasets_stop_on_newline(self) -> None:
        self.assertIn("\n", stop_strings_for_dataset("coqa"))
        self.assertEqual(stop_strings_for_dataset("gsm8k"), [])


if __name__ == "__main__":
    unittest.main()
