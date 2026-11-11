import unittest

from sentiment.fuzzy_model import analyze_review, clean_review, fuzzy_inference


class FuzzySentimentTests(unittest.TestCase):
    def test_fuzzy_output_stays_in_universe(self):
        for polarity, intensity in [(-1, 1), (-0.8, 0.8), (0, 0), (0.8, 0.8), (1, 1)]:
            score = fuzzy_inference(polarity, intensity)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)

    def test_positive_and_negative_polarity_are_ordered(self):
        negative = fuzzy_inference(-0.9, 0.8)
        neutral = fuzzy_inference(0.0, 0.2)
        positive = fuzzy_inference(0.9, 0.8)
        self.assertLess(negative, neutral)
        self.assertLess(neutral, positive)

    def test_html_is_removed(self):
        self.assertEqual(clean_review("Great<br /><b>movie</b>!"), "Great movie !")

    def test_single_review_returns_expected_fields(self):
        result = analyze_review("This movie was absolutely wonderful and moving.")
        self.assertIn("fuzzy_score", result)
        self.assertIn("polarity", result)
        self.assertIn(result["fuzzy_class"], {"positive", "negative", "neutral"})


if __name__ == "__main__":
    unittest.main()
