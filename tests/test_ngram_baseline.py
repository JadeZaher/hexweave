"""Witten-Bell n-gram baseline: order separation on a periodic stream, and per-context normalization."""

import unittest

from scripts import ngram_baseline


class NgramBaselineTest(unittest.TestCase):
    def test_bigram_separates_periodic_stream_from_unigram(self):
        """A fully predictable period-2 stream: bigram should near-eliminate entropy; unigram should not."""
        stream = b"ab" * 2000
        start = bytes((ngram_baseline.START,))
        tables = ngram_baseline.build_tables(start + stream, max_order=2)
        bits_order1, count1 = ngram_baseline.score(tables, 1, start, stream)
        bits_order2, count2 = ngram_baseline.score(tables, 2, start, stream)
        self.assertTrue(0.9 < bits_order1 / count1 < 1.1, f"order-1 bits/byte was {bits_order1 / count1}")
        self.assertLess(bits_order2 / count2, 0.05)

    def test_context_probabilities_sum_to_one(self):
        """Witten-Bell interpolation must leave each context's 256-way distribution properly normalized."""
        tables = ngram_baseline.build_tables(bytes((ngram_baseline.START,)) + b"abcabcabd", max_order=2)
        total = sum(ngram_baseline.probability(byte_value, b"a", 2, tables) for byte_value in range(256))
        self.assertAlmostEqual(total, 1.0, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
