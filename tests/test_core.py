"""Tests for memtrace-diagnostics core modules."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from memtrace.collector import SyntheticTrainingSimulator, MemoryTrace, MemorySnapshot
from memtrace.analyzer import SpectralAnalyzer, ChaosDetector


class TestSyntheticSimulator:
    def setup_method(self):
        self.sim = SyntheticTrainingSimulator(steps=50, seed=42)

    def test_generates_correct_length(self):
        trace = self.sim.generate("stable")
        assert len(trace.snapshots) == 50 * 10  # steps * samples_per_step

    def test_all_regimes_produce_traces(self):
        for regime in ("stable", "transitional", "chaotic"):
            trace = self.sim.generate(regime)
            assert len(trace.snapshots) > 0
            assert all(isinstance(s, MemorySnapshot) for s in trace.snapshots)

    def test_memory_within_bounds(self):
        for regime in ("stable", "transitional", "chaotic"):
            self.sim.rng = np.random.default_rng(hash(regime) % (2**31))
            trace = self.sim.generate(regime)
            mem = trace.memory_mb
            assert mem.min() >= self.sim.base * 0.5
            assert mem.max() <= self.sim.peak * 1.4

    def test_config_preserved(self):
        trace = self.sim.generate("chaotic")
        assert trace.config["regime"] == "chaotic"
        assert trace.config["steps"] == 50

    def test_timestamps_monotonic(self):
        trace = self.sim.generate("stable")
        ts = trace.timestamps
        assert np.all(np.diff(ts) >= 0)


class TestMemoryTraceIO:
    def test_save_load_roundtrip(self):
        sim = SyntheticTrainingSimulator(steps=10, seed=0)
        original = sim.generate("stable")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            original.save(f.name)
            loaded = MemoryTrace.load(f.name)
        assert len(loaded.snapshots) == len(original.snapshots)
        assert loaded.config == original.config
        np.testing.assert_allclose(loaded.memory_mb, original.memory_mb, rtol=1e-6)


class TestSpectralAnalyzer:
    def setup_method(self):
        self.analyzer = SpectralAnalyzer(sampling_rate=10.0)

    def test_pure_sine_has_low_entropy(self):
        t = np.arange(1024) / 10.0
        signal = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz sine
        result = self.analyzer.analyze(signal)
        assert result.spectral_entropy < 3.0

    def test_white_noise_has_high_entropy(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(1024)
        result = self.analyzer.analyze(signal)
        assert result.spectral_entropy > 5.0

    def test_dominant_frequency_detection(self):
        t = np.arange(2048) / 10.0
        signal = 5 * np.sin(2 * np.pi * 2.0 * t) + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        result = self.analyzer.analyze(signal)
        assert abs(result.dominant_freq - 2.0) < 0.1

    def test_bandpass_reconstruction(self):
        t = np.arange(1024) / 10.0
        signal = np.sin(2 * np.pi * 1.0 * t) + np.sin(2 * np.pi * 3.0 * t)
        result = self.analyzer.analyze(signal, bandpass=(0.5, 1.5))
        assert result.signal_filtered is not None
        assert len(result.signal_filtered) == len(signal)

    def test_spectrogram_shape(self):
        signal = np.random.default_rng(0).standard_normal(2000)
        times, freqs, power_db = self.analyzer.spectrogram(signal)
        assert len(times) > 0
        assert len(freqs) > 0
        assert power_db.shape == (len(freqs), len(times))


class TestChaosDetector:
    def setup_method(self):
        self.detector = ChaosDetector(window_size=100)

    def test_stable_trace_low_instability(self):
        sim = SyntheticTrainingSimulator(steps=100, seed=42)
        trace = sim.generate("stable")
        metrics = self.detector.analyze(trace.memory_mb)
        assert metrics.instability_score < 0.6

    def test_chaotic_trace_high_instability(self):
        sim = SyntheticTrainingSimulator(steps=100, seed=42)
        sim.rng = np.random.default_rng(256)
        trace = sim.generate("chaotic")
        metrics = self.detector.analyze(trace.memory_mb)
        assert metrics.instability_score > 0.5

    def test_regime_ordering(self):
        sim = SyntheticTrainingSimulator(steps=150, seed=42)
        scores = {}
        for regime in ("stable", "transitional", "chaotic"):
            sim.rng = np.random.default_rng(hash(regime) % (2**31))
            trace = sim.generate(regime)
            metrics = self.detector.analyze(trace.memory_mb)
            scores[regime] = metrics.instability_score
        assert scores["stable"] < scores["chaotic"]

    def test_hurst_in_range(self):
        signal = np.random.default_rng(0).standard_normal(2000)
        metrics = self.detector.analyze(signal)
        assert 0 <= metrics.hurst_exponent <= 1

    def test_rolling_std_shape(self):
        signal = np.random.default_rng(0).standard_normal(500)
        metrics = self.detector.analyze(signal)
        assert len(metrics.rolling_std) == len(signal)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
