import unittest

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from features import FEATURE_NAMES, canonical_pair, clean_text, make_matrix, pair_features
from predict import predict_pair
from train import prepare_data, split_data


class ProjectTests(unittest.TestCase):
    def test_feature_values_and_empty_text(self):
        basic = pair_features("how to learn python", "how to study python", "basic")
        self.assertEqual(basic[4:], [3, 8, 0.375])
        full = pair_features("", "")
        self.assertEqual(len(full), len(FEATURE_NAMES))
        self.assertEqual(len(full), 22)
        self.assertTrue(np.isfinite(full).all())

    def test_cleaning_preserves_negation_and_language_symbols(self):
        result = clean_text("I can't learn C++ in 2026!")
        self.assertIn("cannot", result)
        self.assertIn("c++", result)
        self.assertIn("2026", result)
        self.assertEqual(clean_text(result), result)

    def test_swapping_questions_does_not_change_features(self):
        data = pd.DataFrame({"question1": ["learn python"], "question2": ["study python"]})
        swapped = data.rename(columns={"question1": "question2", "question2": "question1"})
        vectorizer = CountVectorizer().fit(["learn python", "study python"])
        for mode in ["bow", "basic", "full"]:
            first = make_matrix(data, vectorizer, mode)
            second = make_matrix(swapped, vectorizer, mode)
            self.assertEqual((first != second).nnz, 0)

    def test_duplicate_and_conflicting_pairs_are_removed(self):
        data = pd.DataFrame([
            ["a b", "c d", 1], ["c d", "a b", 1],
            ["x y", "z w", 0], ["z w", "x y", 1],
            [None, "a", 0], ["!!!", "b", 0]
        ], columns=["question1", "question2", "is_duplicate"])
        cleaned = prepare_data(data)
        self.assertEqual(len(cleaned), 1)

    def test_splits_do_not_share_pairs(self):
        data = pd.DataFrame({
            "question1": [f"first question {i}" for i in range(80)],
            "question2": [f"second question {i}" for i in range(80)],
            "is_duplicate": [i % 2 for i in range(80)]
        })
        parts = split_data(prepare_data(data), 42)
        keys = [set(canonical_pair(a, b) for a, b in zip(p.question1, p.question2)) for p in parts]
        self.assertFalse(keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])

    def test_invalid_prediction_inputs(self):
        with self.assertRaises(ValueError):
            predict_pair(" ", "valid question", {})
        with self.assertRaises(ValueError):
            predict_pair("x" * 2001, "valid question", {})


if __name__ == "__main__":
    unittest.main()
