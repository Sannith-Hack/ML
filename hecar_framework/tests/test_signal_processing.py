"""HECAR Framework — Test: Signal Processing"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np


class TestNoiseFilter(unittest.TestCase):

    def setUp(self):
        from modules.signal_processing.noise_filter import NoiseFilter
        self.nf = NoiseFilter()
        t = np.linspace(0, 2, 720)
        self.signal = np.sin(2 * np.pi * 1.5 * t) + 0.3 * np.random.randn(720)

    def test_bandpass_output_shape(self):
        result = self.nf.bandpass_filter(self.signal)
        self.assertEqual(result.shape, self.signal.shape)

    def test_filter_output_shape(self):
        result = self.nf.filter(self.signal)
        self.assertEqual(result.shape, self.signal.shape)


class TestNormalizer(unittest.TestCase):

    def setUp(self):
        from modules.signal_processing.normalizer import Normalizer
        self.norm = Normalizer()
        self.signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_minmax_range(self):
        result = self.norm.minmax_normalize(self.signal)
        self.assertAlmostEqual(result.min(), 0.0, places=5)
        self.assertAlmostEqual(result.max(), 1.0, places=5)

    def test_zscore_mean(self):
        result = self.norm.zscore_normalize(self.signal)
        self.assertAlmostEqual(result.mean(), 0.0, places=5)


class TestSegmenter(unittest.TestCase):

    def setUp(self):
        from modules.signal_processing.segmenter import Segmenter
        self.seg = Segmenter()
        t = np.linspace(0, 5, 1800)
        # Synthetic ECG-like signal with peaks
        self.signal = np.zeros(1800)
        for peak in [180, 540, 900, 1260, 1620]:
            self.signal[peak] = 1.0

    def test_detect_peaks(self):
        peaks = self.seg.detect_qrs_peaks(self.signal)
        self.assertGreater(len(peaks), 0)

    def test_segment_output(self):
        peaks = self.seg.detect_qrs_peaks(self.signal)
        if len(peaks) > 0:
            beats = self.seg.segment_beats(self.signal, peaks)
            self.assertEqual(beats.ndim, 2)


if __name__ == "__main__":
    unittest.main()
