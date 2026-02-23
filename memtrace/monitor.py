"""
Real-time training monitor — wraps collector + analyzer for live alerting.

Designed to be dropped into any PyTorch/TensorFlow training loop as a callback
with near-zero overhead. Accumulates memory snapshots and periodically runs
spectral analysis to flag instability before it shows up in loss.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Literal, Optional

import numpy as np

from memtrace.collector import MemoryCollector, MemoryTrace
from memtrace.analyzer import SpectralAnalyzer, ChaosDetector, ChaosMetrics, SpectralResult

logger = logging.getLogger("memtrace")


class TrainingMonitor:
    """
    Drop-in training loop monitor.

    Usage::

        monitor = TrainingMonitor(backend="cuda", alert_callback=my_fn)
        for step, batch in enumerate(loader):
            monitor.step_start(step)
            loss = model(batch)
            loss.backward()
            optimizer.step()
            alert = monitor.step_end(step, loss=loss.item())
            if alert:
                print(f"⚠ Instability detected at step {step}: {alert}")

    Parameters
    ----------
    backend : str
        "cuda" or "cpu" memory tracking.
    analysis_interval : int
        Run spectral analysis every N steps.
    window_size : int
        Rolling window for chaos metrics (in samples).
    entropy_threshold : float
        Spectral entropy above this triggers alert.
    instability_threshold : float
        Composite instability score above this triggers alert.
    alert_callback : callable or None
        Called with (step, ChaosMetrics) when instability detected.
    """

    def __init__(
        self,
        backend: Literal["cuda", "cpu"] = "cuda",
        analysis_interval: int = 50,
        window_size: int = 200,
        entropy_threshold: float = 5.0,
        instability_threshold: float = 0.6,
        alert_callback: Optional[Callable] = None,
    ):
        self.collector = MemoryCollector(backend=backend)
        self.analyzer = SpectralAnalyzer(sampling_rate=10.0)
        self.chaos_detector = ChaosDetector(
            window_size=window_size,
            entropy_threshold=entropy_threshold,
            instability_threshold=instability_threshold,
        )
        self.analysis_interval = analysis_interval
        self.instab_thresh = instability_threshold
        self.alert_callback = alert_callback

        self._losses: list[float] = []
        self._grad_norms: list[float] = []
        self._latest_chaos: Optional[ChaosMetrics] = None
        self._latest_spectral: Optional[SpectralResult] = None
        self._alerts: list[dict] = []

    def step_start(self, step: int) -> None:
        """Record memory at start of training step."""
        self.collector.record(step, phase="forward")

    def step_end(
        self,
        step: int,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Record memory at end of step; optionally run analysis.

        Returns alert dict if instability detected, else None.
        """
        self.collector.record(step, phase="optimizer")
        if loss is not None:
            self._losses.append(loss)
        if grad_norm is not None:
            self._grad_norms.append(grad_norm)

        # periodic analysis
        if step > 0 and step % self.analysis_interval == 0:
            return self._run_analysis(step)
        return None

    def _run_analysis(self, step: int) -> Optional[dict]:
        trace = self.collector.trace
        mem = trace.memory_mb
        if len(mem) < 64:
            return None

        self._latest_spectral = self.analyzer.analyze(mem)
        self._latest_chaos = self.chaos_detector.analyze(mem)

        if self._latest_chaos.instability_score > self.instab_thresh:
            alert = {
                "step": step,
                "instability_score": self._latest_chaos.instability_score,
                "spectral_entropy": self._latest_spectral.spectral_entropy,
                "hurst_exponent": self._latest_chaos.hurst_exponent,
                "message": (
                    f"Memory instability detected (score={self._latest_chaos.instability_score:.3f}, "
                    f"spectral_entropy={self._latest_spectral.spectral_entropy:.2f})"
                ),
            }
            self._alerts.append(alert)
            logger.warning(alert["message"])
            if self.alert_callback:
                self.alert_callback(step, self._latest_chaos)
            return alert
        return None

    @property
    def trace(self) -> MemoryTrace:
        return self.collector.trace

    @property
    def chaos_metrics(self) -> Optional[ChaosMetrics]:
        return self._latest_chaos

    @property
    def spectral_result(self) -> Optional[SpectralResult]:
        return self._latest_spectral

    @property
    def alerts(self) -> list[dict]:
        return self._alerts

    def summary(self) -> dict:
        """Return a summary dict for logging / dashboard integration."""
        s = {
            "total_snapshots": len(self.collector.trace.snapshots),
            "total_alerts": len(self._alerts),
        }
        if self._latest_chaos:
            s["instability_score"] = self._latest_chaos.instability_score
            s["hurst_exponent"] = self._latest_chaos.hurst_exponent
            s["sample_entropy"] = self._latest_chaos.sample_entropy
        if self._latest_spectral:
            s["spectral_entropy"] = self._latest_spectral.spectral_entropy
            s["dominant_freq_hz"] = self._latest_spectral.dominant_freq
        return s
