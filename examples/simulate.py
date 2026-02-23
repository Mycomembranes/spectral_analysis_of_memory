#!/usr/bin/env python3
"""
memtrace example: Generate synthetic training traces and run full analysis.

Produces JSON results suitable for the interactive dashboard.

Usage:
    python -m examples.simulate
    python examples/simulate.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memtrace.collector import SyntheticTrainingSimulator
from memtrace.analyzer import SpectralAnalyzer, ChaosDetector


def run_simulation(
    steps: int = 300,
    seed: int = 42,
    output_dir: str = "results",
) -> dict:
    """Run full simulation across all three regimes and return analysis."""

    sim = SyntheticTrainingSimulator(
        base_memory_mb=2000,
        peak_memory_mb=8000,
        steps=steps,
        batch_size=32,
        samples_per_step=10,
        seed=seed,
    )
    analyzer = SpectralAnalyzer(sampling_rate=10.0)
    chaos_det = ChaosDetector(window_size=200, entropy_threshold=5.0)

    results = {}

    for regime in ("stable", "transitional", "chaotic"):
        print(f"\n{'='*60}")
        print(f"  Regime: {regime.upper()}")
        print(f"{'='*60}")

        # reset RNG per regime for determinism
        sim.rng = np.random.default_rng(hash(regime) % (2**31))
        trace = sim.generate(regime)
        mem = trace.memory_mb

        # spectral analysis
        spectral = analyzer.analyze(mem)
        print(f"  Spectral entropy:  {spectral.spectral_entropy:.3f}")
        print(f"  Dominant freq:     {spectral.dominant_freq:.4f} Hz")
        print(f"  Dominant period:   {spectral.dominant_period:.2f} s")
        print(f"  90% bandwidth:     {spectral.bandwidth_90:.4f} Hz")

        # chaos metrics
        chaos = chaos_det.analyze(mem, sampling_rate=10.0)
        print(f"  Hurst exponent:    {chaos.hurst_exponent:.3f}")
        print(f"  Sample entropy:    {chaos.sample_entropy:.3f}")
        print(f"  Instability score: {chaos.instability_score:.3f}")
        print(f"  Unstable regions:  {len(chaos.instability_regions)}")

        # spectrogram
        times, freqs, power_db = analyzer.spectrogram(mem, window_size=256, hop=64)

        # downsample for JSON output (keep every 10th point for memory trace)
        ds = 5
        results[regime] = {
            "timestamps": trace.timestamps[::ds].tolist(),
            "memory_mb": mem[::ds].tolist(),
            "fft_frequencies": spectral.frequencies[:200].tolist(),
            "fft_power": spectral.power[:200].tolist(),
            "spectral_entropy": float(spectral.spectral_entropy),
            "dominant_freq": float(spectral.dominant_freq),
            "dominant_period": float(spectral.dominant_period),
            "bandwidth_90": float(spectral.bandwidth_90),
            "hurst_exponent": float(chaos.hurst_exponent),
            "sample_entropy": float(chaos.sample_entropy),
            "instability_score": float(chaos.instability_score),
            "rolling_std": chaos.rolling_std[::ds].tolist(),
            "rolling_spectral_entropy": [
                float(x) if not np.isnan(x) else None
                for x in chaos.rolling_spectral_entropy[::ds]
            ],
            "instability_regions": chaos.instability_regions,
            "spectrogram_times": times.tolist(),
            "spectrogram_freqs": freqs[:50].tolist(),
            "spectrogram_power_db": power_db[:50].tolist(),
        }

    # save
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    outfile = out / "simulation_results.json"
    with open(outfile, "w") as f:
        json.dump(results, f)
    print(f"\nResults saved to {outfile}")
    return results


if __name__ == "__main__":
    run_simulation()
