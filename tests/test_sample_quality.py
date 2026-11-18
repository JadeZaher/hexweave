"""Sample-quality metrics: loop detection, per-sample shares and the three-occurrence vocabulary."""

import tempfile
import unittest
from pathlib import Path

from scripts import sample_quality


class SampleQualityTest(unittest.TestCase):
    def test_metrics_match_the_reference_definitions(self):
        """Loops need a full 64-byte tail with period <= 32; shares count lower-case words and 4-grams."""
        for data, looping in [(b"ab" * 40, True), (bytes(range(33)) * 2, False), (b"ab" * 31, False),
                              (b"x" * 10 + bytes(range(32)) * 2, True)]:
            self.assertEqual(sample_quality.loops(data), looping, data)
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "train.txt"
            train.write_bytes(b"The cat sat.\nthe CAT ran\nthe cat hid\ndog dog\n")
            words = sample_quality.vocabulary(train)
        self.assertEqual(words, {"the", "cat"})
        result = sample_quality.metrics(b"The cat saw a dog", words)
        self.assertEqual(result["known_words"], 2 / 5)
        self.assertEqual(result["mean_word_length"], 13 / 5)
        self.assertEqual(result["distinct_4grams"], 1.0)
        self.assertFalse(result["loops"])

    def test_fingerprint_check_needs_every_hash_of_the_report_shape(self):
        """Core reports carry state hashes only; connected ones must also carry checkpoint hashes."""
        state = {"before_sha256": "a", "after_sha256": "a"}
        both = state | {"checkpoint_before_sha256": "c", "checkpoint_after_sha256": "c"}
        self.assertEqual(sample_quality.fingerprint(state, connected=False), "a")
        self.assertEqual(sample_quality.fingerprint(both, connected=True), "a")
        for report, connected in [(state, True), ({}, False), (state | {"after_sha256": "b"}, False)]:
            with self.assertRaises(SystemExit):
                sample_quality.fingerprint(report, connected)


if __name__ == "__main__":
    unittest.main()
