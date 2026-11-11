"""The optional NLTK package must not require an online corpus download."""

import unittest
from unittest.mock import patch

from analyzer import GamingSpendingAnalyzer


class SentimentFallbackTests(unittest.TestCase):
    def test_missing_vader_lexicon_still_scores_sentiment(self):
        with patch("analyzer.NLTK_AVAILABLE", True), patch(
            "analyzer.SentimentIntensityAnalyzer", side_effect=LookupError("vader_lexicon missing")
        ):
            analyzer = GamingSpendingAnalyzer()
        score, label = analyzer._analyze_sentiment("I love this awesome game, great value!")
        self.assertGreater(score, 0)
        self.assertEqual(label, "positive")

    def test_without_nltk_still_scores_sentiment(self):
        with patch("analyzer.NLTK_AVAILABLE", False):
            analyzer = GamingSpendingAnalyzer()
        score, label = analyzer._analyze_sentiment("I regret this terrible waste of money")
        self.assertLess(score, 0)
        self.assertEqual(label, "negative")


if __name__ == "__main__":
    unittest.main()
