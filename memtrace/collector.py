"""
Memory trace collection — real hardware logging and synthetic simulation.

The SyntheticTrainingSimulator generates realistic memory traces for three
regimes (stable, transitional, chaotic) by modeling the deterministic
sawtooth of forward/backward/optimizer steps plus stochastic perturbations
that mimic gradient-norm explosions, allocator fragmentation, and
optimizer-state drift.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MemorySnapshot:
    """Single time-point measurement."""
    timestamp: float          # seconds since collection start
    step: int                 # training step index
    memory_bytes: int         # allocated memory in bytes
    phase: str = "unknown"    # forward | backward | optimizer | idle
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryTrace:
    """Full time-series of memory snapshots."""
    snapshots: list[MemorySnapshot] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    # -- convenience arrays --------------------------------------------------
    @property
    def timestamps(self) -> np.ndarray:
        return np.array([s.timestamp for s in self.snapshots])

    @property
    def memory_mb(self) -> np.ndarray:
        return np.array([s.memory_bytes for s in self.snapshots]) / 1e6

    @property
    def steps(self) -> np.ndarray:
        return np.array([s.step for s in self.snapshots])

    # -- I/O -----------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "config": self.config,
            "snapshots": [asdict(s) for s in self.snapshots],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MemoryTrace":
        data = json.loads(Path(path).read_text())
        snaps = [MemorySnapshot(**s) for s in data["snapshots"]]
        return cls(snapshots=snaps, config=data.get("config", {}))


# ---------------------------------------------------------------------------
# Real hardware collector (GPU or CPU)
# ---------------------------------------------------------------------------

class MemoryCollector:
    """
    Lightweight callback-style collector for real training loops.

    Usage (PyTorch)::

        collector = MemoryCollector(backend="cuda")
        for step, batch in enumerate(loader):
            collector.record(step, phase="forward")
            loss = model(batch)
            collector.record(step, phase="backward")
            loss.backward()
            collector.record(step, phase="optimizer")
            optimizer.step()
        trace = collector.trace
    """

    def __init__(self, backend: Literal["cuda", "cpu"] = "cuda"):
        self.backend = backend
        self._start = time.monotonic()
        self._trace = MemoryTrace(config={"backend": backend})

    def _read_memory(self) -> int:
        if self.backend == "cuda":
            try:
                import torch
                return torch.cuda.memory_allocated()
            except Exception:
                return 0
        else:
            try:
                import psutil
                return psutil.Process().memory_info().rss
            except Exception:
                return 0

    def record(self, step: int, phase: str = "unknown", **meta) -> None:
        snap = MemorySnapshot(
            timestamp=time.monotonic() - self._start,
            step=step,
            memory_bytes=self._read_memory(),
            phase=phase,
            metadata=meta,
        )
        self._trace.snapshots.append(snap)

    @property
    def trace(self) -> MemoryTrace:
        return self._trace


# ---------------------------------------------------------------------------
# Synthetic training simulator
# ---------------------------------------------------------------------------

class SyntheticTrainingSimulator:
    """
    Generates physically-motivated synthetic VRAM traces for three regimes.

    Each training step is modelled as three phases (forward → backward → optimizer)
    with a deterministic sawtooth baseline plus regime-dependent noise:

    - **stable**: low-variance Gaussian noise, clean periodic signal
    - **transitional**: growing variance with intermittent spikes (gradient bursts)
    - **chaotic**: heavy-tailed noise, correlated drifts, fragmentation events

    Parameters
    ----------
    base_memory_mb : float
        Resting (idle) VRAM in MB.
    peak_memory_mb : float
        Peak forward-pass VRAM in MB.
    steps : int
        Total training steps to simulate.
    batch_size : int
        Batch size (affects sawtooth amplitude scaling).
    samples_per_step : int
        Temporal sub-samples per training step (resolution).
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        base_memory_mb: float = 2000.0,
        peak_memory_mb: float = 8000.0,
        steps: int = 500,
        batch_size: int = 32,
        samples_per_step: int = 10,
        seed: Optional[int] = None,
    ):
        self.base = base_memory_mb
        self.peak = peak_memory_mb
        self.steps = steps
        self.batch_size = batch_size
        self.sps = samples_per_step
        self.rng = np.random.default_rng(seed)

    # -- internal helpers ---------------------------------------------------

    def _sawtooth_step(self, step: int) -> np.ndarray:
        """Deterministic forward/backward/optimizer sawtooth for one step."""
        t = np.linspace(0, 1, self.sps)
        # forward ramp-up (0→0.4), backward decay (0.4→0.8), optimizer bump (0.8→1)
        mem = np.piecewise(
            t,
            [t < 0.4, (t >= 0.4) & (t < 0.8), t >= 0.8],
            [
                lambda x: self.base + (self.peak - self.base) * (x / 0.4),
                lambda x: self.peak - (self.peak - self.base) * 0.7 * ((x - 0.4) / 0.4),
                lambda x: self.base + (self.peak - self.base) * 0.3
                         + (self.peak - self.base) * 0.05 * np.sin(10 * np.pi * (x - 0.8)),
            ],
        )
        return mem

    def _phase_labels(self) -> list[str]:
        t = np.linspace(0, 1, self.sps)
        labels = []
        for ti in t:
            if ti < 0.4:
                labels.append("forward")
            elif ti < 0.8:
                labels.append("backward")
            else:
                labels.append("optimizer")
        return labels

    # -- regime noise models ------------------------------------------------

    def _stable_noise(self, n: int) -> np.ndarray:
        sigma = (self.peak - self.base) * 0.01
        return self.rng.normal(0, sigma, n)

    def _transitional_noise(self, step: int, n: int) -> np.ndarray:
        progress = step / max(self.steps, 1)
        sigma = (self.peak - self.base) * (0.01 + 0.08 * progress)
        noise = self.rng.normal(0, sigma, n)
        # intermittent gradient spikes
        if self.rng.random() < 0.06 * (1 + 3 * progress):
            spike_idx = self.rng.integers(0, n)
            noise[spike_idx] += self.rng.exponential(sigma * 6)
        return noise

    def _chaotic_noise(self, step: int, n: int) -> np.ndarray:
        scale = (self.peak - self.base) * 0.15
        # heavy-tailed (Student-t with low df)
        noise = self.rng.standard_t(df=3, size=n) * scale
        # correlated drift (random walk component)
        drift = np.cumsum(self.rng.normal(0, scale * 0.1, n))
        # fragmentation events (sudden drops/spikes)
        frag = np.zeros(n)
        for _ in range(self.rng.poisson(1.5)):
            idx = self.rng.integers(0, n)
            frag[idx] = self.rng.choice([-1, 1]) * self.rng.exponential(scale * 3)
        return noise + drift + frag

    # -- main generators ----------------------------------------------------

    def generate(
        self,
        regime: Literal["stable", "transitional", "chaotic"] = "stable",
    ) -> MemoryTrace:
        """Generate a full memory trace for the given regime."""
        snapshots: list[MemorySnapshot] = []
        phase_labels = self._phase_labels()
        dt = 0.1  # seconds per sub-sample

        for step in range(self.steps):
            baseline = self._sawtooth_step(step)

            if regime == "stable":
                noise = self._stable_noise(self.sps)
            elif regime == "transitional":
                noise = self._transitional_noise(step, self.sps)
            else:
                noise = self._chaotic_noise(step, self.sps)

            mem = np.clip(baseline + noise, self.base * 0.5, self.peak * 1.4)

            for i in range(self.sps):
                snapshots.append(MemorySnapshot(
                    timestamp=step * self.sps * dt + i * dt,
                    step=step,
                    memory_bytes=int(mem[i] * 1e6),
                    phase=phase_labels[i],
                ))

        return MemoryTrace(
            snapshots=snapshots,
            config={
                "regime": regime,
                "steps": self.steps,
                "base_mb": self.base,
                "peak_mb": self.peak,
                "batch_size": self.batch_size,
                "samples_per_step": self.sps,
            },
        )

    def generate_all(self) -> dict[str, MemoryTrace]:
        """Generate traces for all three regimes (same seed/params)."""
        results = {}
        for regime in ("stable", "transitional", "chaotic"):
            # re-seed to keep deterministic per regime
            self.rng = np.random.default_rng(
                hash(regime) % (2**31) if self.rng is not None else None
            )
            results[regime] = self.generate(regime)
        return results
