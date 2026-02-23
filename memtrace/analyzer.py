"""
Spectral analysis and chaos-detection metrics for memory traces.

Implements:
- FFT decomposition with power spectral density
- Spectral entropy (normalized Shannon entropy of PSD)
- Rolling chaos metrics (std-dev, Hurst exponent, sample entropy)
- Dominant-frequency extraction and band-pass filtering
- Instability detection with configurable thresholds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.fft import rfft, rfftfreq, irfft


@dataclass
class SpectralResult:
    """Container for FFT analysis results."""
    frequencies: np.ndarray       # Hz
    power: np.ndarray             # |FFT|^2, normalized
    spectral_entropy: float       # 0 = pure tone, log(N) = white noise
    dominant_freq: float          # Hz — strongest periodic component
    dominant_period: float        # seconds — 1/dominant_freq
    bandwidth_90: float           # freq range containing 90% power
    signal_filtered: Optional[np.ndarray] = None  # band-pass reconstructed


@dataclass
class ChaosMetrics:
    """Rolling chaos statistics over a memory trace."""
    rolling_std: np.ndarray           # windowed standard deviation
    rolling_spectral_entropy: np.ndarray  # windowed spectral entropy
    hurst_exponent: float             # 0.5=random walk, <0.5=mean-revert, >0.5=trending
    sample_entropy: float             # regularity measure (lower=more regular)
    instability_score: float          # composite 0-1 score
    instability_regions: list[tuple[int, int]] = field(default_factory=list)  # (start, end) indices


class SpectralAnalyzer:
    """
    FFT-based spectral analysis of memory time-series.

    Parameters
    ----------
    sampling_rate : float
        Samples per second (Hz). Default 10 Hz matches nvidia-smi -l 0.1.
    """

    def __init__(self, sampling_rate: float = 10.0):
        self.fs = sampling_rate

    def analyze(
        self,
        signal: np.ndarray,
        bandpass: Optional[tuple[float, float]] = None,
    ) -> SpectralResult:
        """
        Full spectral decomposition.

        Parameters
        ----------
        signal : array
            1-D memory usage time-series (e.g., MB).
        bandpass : (low_hz, high_hz) or None
            If given, also compute inverse-FFT reconstruction of that band.
        """
        # detrend (remove DC / linear drift)
        n = len(signal)
        x = signal - np.mean(signal)
        trend = np.polyval(np.polyfit(np.arange(n), signal, 1), np.arange(n))
        x = signal - trend

        # windowing (Hann) to reduce spectral leakage
        window = np.hanning(n)
        x_w = x * window

        # FFT
        fft_vals = rfft(x_w)
        freqs = rfftfreq(n, d=1.0 / self.fs)
        power = np.abs(fft_vals) ** 2
        power[0] = 0  # ignore DC

        # normalize to probability distribution for entropy
        total = power.sum()
        if total < 1e-12:
            p = np.ones_like(power) / len(power)
        else:
            p = power / total

        # spectral entropy (Shannon)
        p_safe = p[p > 0]
        spectral_entropy = -np.sum(p_safe * np.log2(p_safe))

        # dominant frequency
        dom_idx = np.argmax(power[1:]) + 1  # skip DC
        dominant_freq = freqs[dom_idx]
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else np.inf

        # 90% bandwidth
        cumpower = np.cumsum(np.sort(power)[::-1]) / total if total > 0 else np.ones(1)
        n90 = np.searchsorted(cumpower, 0.9) + 1
        sorted_freqs = freqs[np.argsort(power)[::-1]]
        bandwidth_90 = float(sorted_freqs[:n90].max() - sorted_freqs[:n90].min()) if n90 > 1 else 0.0

        # optional band-pass filter + inverse FFT
        filtered = None
        if bandpass is not None:
            lo, hi = bandpass
            mask = (freqs >= lo) & (freqs <= hi)
            fft_filtered = np.zeros_like(fft_vals)
            fft_filtered[mask] = fft_vals[mask]
            filtered = irfft(fft_filtered, n=n)

        return SpectralResult(
            frequencies=freqs,
            power=power,
            spectral_entropy=spectral_entropy,
            dominant_freq=dominant_freq,
            dominant_period=dominant_period,
            bandwidth_90=bandwidth_90,
            signal_filtered=filtered,
        )

    def spectrogram(
        self,
        signal: np.ndarray,
        window_size: int = 256,
        hop: int = 64,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Short-time FFT spectrogram.

        Returns (times, freqs, power_db) for heatmap visualization.
        """
        n = len(signal)
        freqs = rfftfreq(window_size, d=1.0 / self.fs)
        times = []
        spectra = []

        for start in range(0, n - window_size + 1, hop):
            chunk = signal[start:start + window_size]
            chunk = chunk - np.mean(chunk)
            chunk *= np.hanning(window_size)
            spec = np.abs(rfft(chunk)) ** 2
            spectra.append(spec)
            times.append((start + window_size / 2) / self.fs)

        power = np.array(spectra).T
        # dB scale (floor at -80 dB)
        power_db = 10 * np.log10(np.maximum(power, 1e-8))

        return np.array(times), freqs, power_db


class ChaosDetector:
    """
    Computes rolling and aggregate chaos metrics for instability detection.

    Parameters
    ----------
    window_size : int
        Number of samples for rolling computations.
    entropy_threshold : float
        Spectral entropy above this flags potential instability.
    instability_threshold : float
        Composite score above this triggers alerts.
    """

    def __init__(
        self,
        window_size: int = 200,
        entropy_threshold: float = 5.0,
        instability_threshold: float = 0.6,
    ):
        self.window = window_size
        self.entropy_thresh = entropy_threshold
        self.instab_thresh = instability_threshold

    def analyze(
        self,
        signal: np.ndarray,
        sampling_rate: float = 10.0,
    ) -> ChaosMetrics:
        """Compute all chaos metrics for a memory trace."""
        n = len(signal)
        analyzer = SpectralAnalyzer(sampling_rate)

        # --- rolling standard deviation ---
        rolling_std = self._rolling_stat(signal, np.std)

        # --- rolling spectral entropy ---
        roll_ent = np.full(n, np.nan)
        half_w = self.window // 2
        for i in range(half_w, n - half_w):
            chunk = signal[i - half_w:i + half_w]
            if len(chunk) < 32:
                continue
            res = analyzer.analyze(chunk)
            roll_ent[i] = res.spectral_entropy
        # fill edges
        first_valid = np.where(~np.isnan(roll_ent))[0]
        if len(first_valid) > 0:
            roll_ent[:first_valid[0]] = roll_ent[first_valid[0]]
            last_valid = np.where(~np.isnan(roll_ent))[0][-1]
            roll_ent[last_valid + 1:] = roll_ent[last_valid]

        # --- Hurst exponent (R/S analysis) ---
        hurst = self._hurst_rs(signal)

        # --- sample entropy ---
        samp_ent = self._sample_entropy(signal, m=2, r_frac=0.2)

        # --- composite instability score (0–1) ---
        # blend of normalized entropy + variance + anti-persistence
        global_res = analyzer.analyze(signal)
        max_entropy = np.log2(len(global_res.frequencies))
        norm_entropy = global_res.spectral_entropy / max_entropy if max_entropy > 0 else 0

        cv = np.std(signal) / np.mean(signal) if np.mean(signal) > 0 else 0
        norm_cv = min(cv / 0.3, 1.0)  # CV > 0.3 saturates

        hurst_penalty = abs(hurst - 0.5) * 2  # deviation from random walk
        instability_score = float(np.clip(
            0.45 * norm_entropy + 0.35 * norm_cv + 0.20 * (1 - hurst_penalty),
            0, 1,
        ))

        # --- detect instability regions ---
        regions = []
        if roll_ent is not None and not np.all(np.isnan(roll_ent)):
            above = roll_ent > self.entropy_thresh
            changes = np.diff(above.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1
            if above[0]:
                starts = np.concatenate(([0], starts))
            if above[-1]:
                ends = np.concatenate((ends, [n - 1]))
            for s, e in zip(starts, ends):
                regions.append((int(s), int(e)))

        return ChaosMetrics(
            rolling_std=rolling_std,
            rolling_spectral_entropy=roll_ent,
            hurst_exponent=hurst,
            sample_entropy=samp_ent,
            instability_score=instability_score,
            instability_regions=regions,
        )

    # -- internal -----------------------------------------------------------

    def _rolling_stat(self, signal: np.ndarray, fn) -> np.ndarray:
        n = len(signal)
        out = np.full(n, np.nan)
        half = self.window // 2
        for i in range(half, n - half):
            out[i] = fn(signal[i - half:i + half])
        if n > 0:
            first_valid = np.where(~np.isnan(out))[0]
            if len(first_valid) > 0:
                out[:first_valid[0]] = out[first_valid[0]]
                lv = np.where(~np.isnan(out))[0][-1]
                out[lv + 1:] = out[lv]
        return out

    @staticmethod
    def _hurst_rs(signal: np.ndarray, min_window: int = 20) -> float:
        """Rescaled-range (R/S) Hurst exponent estimate."""
        n = len(signal)
        if n < min_window * 4:
            return 0.5  # fallback

        max_k = int(np.log2(n / min_window))
        if max_k < 2:
            return 0.5

        rs_list = []
        ns_list = []

        for k in range(1, max_k + 1):
            size = int(n / (2 ** k))
            if size < min_window:
                break
            rs_vals = []
            for start in range(0, n - size + 1, size):
                chunk = signal[start:start + size]
                mean = chunk.mean()
                devs = np.cumsum(chunk - mean)
                r = devs.max() - devs.min()
                s = chunk.std()
                if s > 0:
                    rs_vals.append(r / s)
            if rs_vals:
                rs_list.append(np.mean(rs_vals))
                ns_list.append(size)

        if len(rs_list) < 2:
            return 0.5

        log_n = np.log(ns_list)
        log_rs = np.log(rs_list)
        hurst = float(np.polyfit(log_n, log_rs, 1)[0])
        return np.clip(hurst, 0.0, 1.0)

    @staticmethod
    def _sample_entropy(signal: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
        """Approximate sample entropy (regularity measure)."""
        n = len(signal)
        if n < 100:
            return 0.0
        # downsample for speed if very long
        if n > 3000:
            signal = signal[::n // 3000]
            n = len(signal)

        r = r_frac * np.std(signal)
        if r < 1e-12:
            return 0.0

        def _count_matches(length: int) -> int:
            templates = np.array([signal[i:i + length] for i in range(n - length)])
            count = 0
            for i in range(len(templates)):
                dists = np.max(np.abs(templates[i] - templates[i + 1:]), axis=1)
                count += np.sum(dists < r)
            return count

        a = _count_matches(m + 1)
        b = _count_matches(m)

        if b == 0:
            return 0.0
        return -np.log(a / b) if a > 0 else 0.0
