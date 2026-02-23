"""
Cross-correlation analysis between memory spectral features and weight-space dynamics.

Core hypothesis tests:
1. Does memory spectral entropy correlate with gradient norm variance?
2. Does memory chaos *lead* weight-space instability (predictive power)?
3. Can frequency bands in the memory signal be mapped to specific layers?
4. Does injected instability show up in memory before loss?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from memtrace.analyzer import SpectralAnalyzer, ChaosDetector


@dataclass
class CorrelationResult:
    """Result of cross-correlation between two signals."""
    pearson_r: float
    spearman_rho: float
    lag_at_peak: int        # positive = signal_a leads signal_b
    peak_xcorr: float       # cross-correlation at optimal lag
    granger_f_stat: float   # Granger causality F-statistic
    granger_p_value: float  # p-value for Granger test
    mutual_info: float      # mutual information estimate


@dataclass
class MemoryWeightCorrelation:
    """Full correlation analysis between memory and weight dynamics."""
    # global correlations
    memory_vs_grad_norm: CorrelationResult
    memory_vs_loss: CorrelationResult
    memory_vs_param_velocity: CorrelationResult
    memory_vs_adam_v_norm: CorrelationResult

    # spectral correlations
    memory_entropy_vs_grad_variance: CorrelationResult
    memory_entropy_vs_weight_norm_change: CorrelationResult

    # per-layer correlations (name → result)
    layer_correlations: dict[str, CorrelationResult]

    # detection performance
    instability_detection_lag: int  # steps memory leads loss spike
    memory_leads_loss: bool
    memory_leads_gradients: bool

    # information content
    memory_predictive_r2: float  # R² of memory features predicting loss
    gradient_predictive_r2: float  # R² of gradient features predicting loss
    memory_adds_information: bool  # memory improves over gradients alone


class SignalCorrelator:
    """
    Computes cross-correlations and tests predictive relationships
    between memory signals and weight-space dynamics.
    """

    def __init__(self, window_size: int = 50):
        self.window = window_size
        self.analyzer = SpectralAnalyzer(sampling_rate=1.0)  # 1 sample per step

    def correlate_signals(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        max_lag: int = 30,
    ) -> CorrelationResult:
        """Full cross-correlation analysis between two time series."""
        # ensure same length
        n = min(len(signal_a), len(signal_b))
        a = signal_a[:n].copy()
        b = signal_b[:n].copy()

        # handle constant signals
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return CorrelationResult(0, 0, 0, 0, 0, 1.0, 0)

        # z-score normalize
        a = (a - np.mean(a)) / (np.std(a) + 1e-12)
        b = (b - np.mean(b)) / (np.std(b) + 1e-12)

        # Pearson correlation
        pearson_r = float(np.corrcoef(a, b)[0, 1])

        # Spearman rank correlation
        rank_a = np.argsort(np.argsort(a)).astype(float)
        rank_b = np.argsort(np.argsort(b)).astype(float)
        spearman_rho = float(np.corrcoef(rank_a, rank_b)[0, 1])

        # Cross-correlation with lags
        xcorr = np.correlate(a, b, mode="full") / n
        mid = n - 1
        lag_range = slice(max(0, mid - max_lag), min(len(xcorr), mid + max_lag + 1))
        xcorr_window = xcorr[lag_range]
        lags = np.arange(-max_lag, max_lag + 1)[:len(xcorr_window)]
        peak_idx = np.argmax(np.abs(xcorr_window))
        lag_at_peak = int(lags[peak_idx])
        peak_xcorr = float(xcorr_window[peak_idx])

        # Granger causality (simple VAR(1) F-test)
        granger_f, granger_p = self._granger_test(a, b, max_lag=5)

        # Mutual information (binned estimate)
        mi = self._mutual_info(a, b, bins=20)

        return CorrelationResult(
            pearson_r=pearson_r,
            spearman_rho=spearman_rho,
            lag_at_peak=lag_at_peak,
            peak_xcorr=peak_xcorr,
            granger_f_stat=granger_f,
            granger_p_value=granger_p,
            mutual_info=mi,
        )

    def full_analysis(
        self,
        timeseries: dict[str, np.ndarray],
        layer_names: list[str],
    ) -> MemoryWeightCorrelation:
        """
        Run complete correlation analysis between memory and weight dynamics.

        Parameters
        ----------
        timeseries : dict
            Output from WeightDynamicsSimulator.get_timeseries()
        layer_names : list[str]
            Names of layers to analyze individually.
        """
        mem = timeseries["memory_noisy"]
        loss = timeseries["loss"]
        grad = timeseries["total_grad_norm"]
        vel = timeseries["param_velocity"]
        adam_v = timeseries["adam_v_norm"]

        # rolling spectral entropy of memory
        mem_entropy = self._rolling_spectral_entropy(mem)
        grad_var = self._rolling_variance(grad)
        wn_change = self._rolling_variance(timeseries["total_weight_norm"])

        # global correlations
        mem_vs_grad = self.correlate_signals(mem, grad)
        mem_vs_loss = self.correlate_signals(mem, loss)
        mem_vs_vel = self.correlate_signals(mem, vel)
        mem_vs_adam = self.correlate_signals(mem, adam_v)

        # spectral correlations
        ent_vs_gvar = self.correlate_signals(mem_entropy, grad_var)
        ent_vs_wn = self.correlate_signals(mem_entropy, wn_change)

        # per-layer
        layer_corrs = {}
        for name in layer_names:
            gn_key = f"grad_norm_{name}"
            if gn_key in timeseries:
                layer_corrs[name] = self.correlate_signals(mem, timeseries[gn_key])

        # detection lag: does memory lead loss?
        mem_vs_loss_lag = mem_vs_loss.lag_at_peak
        leads_loss = mem_vs_loss_lag > 0
        leads_grad = mem_vs_grad.lag_at_peak > 0

        # predictive R²: can memory features predict future loss?
        mem_pred_r2 = self._predictive_r2(mem, loss, lag=5)
        grad_pred_r2 = self._predictive_r2(grad, loss, lag=5)
        adds_info = mem_pred_r2 > grad_pred_r2 * 0.5  # memory adds if >50% as good

        return MemoryWeightCorrelation(
            memory_vs_grad_norm=mem_vs_grad,
            memory_vs_loss=mem_vs_loss,
            memory_vs_param_velocity=mem_vs_vel,
            memory_vs_adam_v_norm=mem_vs_adam,
            memory_entropy_vs_grad_variance=ent_vs_gvar,
            memory_entropy_vs_weight_norm_change=ent_vs_wn,
            layer_correlations=layer_corrs,
            instability_detection_lag=abs(mem_vs_loss_lag),
            memory_leads_loss=leads_loss,
            memory_leads_gradients=leads_grad,
            memory_predictive_r2=mem_pred_r2,
            gradient_predictive_r2=grad_pred_r2,
            memory_adds_information=adds_info,
        )

    # -- rolling metrics ----------------------------------------------------

    def _rolling_spectral_entropy(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal)
        entropy = np.full(n, np.nan)
        half = self.window // 2
        for i in range(half, n - half):
            chunk = signal[i - half:i + half]
            result = self.analyzer.analyze(chunk)
            entropy[i] = result.spectral_entropy
        # fill edges
        valid = np.where(~np.isnan(entropy))[0]
        if len(valid) > 0:
            entropy[:valid[0]] = entropy[valid[0]]
            entropy[valid[-1] + 1:] = entropy[valid[-1]]
        return entropy

    def _rolling_variance(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal)
        var = np.full(n, np.nan)
        half = self.window // 2
        for i in range(half, n - half):
            var[i] = np.var(signal[i - half:i + half])
        valid = np.where(~np.isnan(var))[0]
        if len(valid) > 0:
            var[:valid[0]] = var[valid[0]]
            var[valid[-1] + 1:] = var[valid[-1]]
        return var

    # -- statistical tests --------------------------------------------------

    @staticmethod
    def _granger_test(
        x: np.ndarray, y: np.ndarray, max_lag: int = 5,
    ) -> tuple[float, float]:
        """
        Simplified Granger causality: does x help predict y beyond y's own history?
        Returns (F-statistic, p-value).
        """
        n = len(x)
        if n < max_lag * 3:
            return 0.0, 1.0

        # restricted model: y_t = Σ α_i * y_{t-i}
        # unrestricted: y_t = Σ α_i * y_{t-i} + Σ β_i * x_{t-i}
        Y = y[max_lag:]
        n_eff = len(Y)

        # build lag matrices
        Y_lags = np.column_stack([y[max_lag - i - 1:n - i - 1] for i in range(max_lag)])
        X_lags = np.column_stack([x[max_lag - i - 1:n - i - 1] for i in range(max_lag)])

        # restricted OLS
        try:
            beta_r = np.linalg.lstsq(Y_lags, Y, rcond=None)[0]
            resid_r = Y - Y_lags @ beta_r
            ssr_r = np.sum(resid_r ** 2)

            # unrestricted OLS
            XY_lags = np.column_stack([Y_lags, X_lags])
            beta_u = np.linalg.lstsq(XY_lags, Y, rcond=None)[0]
            resid_u = Y - XY_lags @ beta_u
            ssr_u = np.sum(resid_u ** 2)

            # F-test
            df1 = max_lag
            df2 = n_eff - 2 * max_lag
            if df2 <= 0 or ssr_u < 1e-12:
                return 0.0, 1.0

            f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)

            # approximate p-value using F-distribution survival function
            # (simplified — for exact, use scipy.stats.f)
            p_value = max(0.0, 1.0 - min(f_stat / (f_stat + df2 / df1), 1.0))

            return float(f_stat), float(p_value)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0, 1.0

    @staticmethod
    def _mutual_info(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
        """Binned mutual information estimate."""
        hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
        hist_xy = hist_xy / hist_xy.sum()
        hist_x = hist_xy.sum(axis=1)
        hist_y = hist_xy.sum(axis=0)

        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if hist_xy[i, j] > 0 and hist_x[i] > 0 and hist_y[j] > 0:
                    mi += hist_xy[i, j] * np.log2(
                        hist_xy[i, j] / (hist_x[i] * hist_y[j])
                    )
        return float(max(mi, 0.0))

    @staticmethod
    def _predictive_r2(
        predictor: np.ndarray,
        target: np.ndarray,
        lag: int = 5,
    ) -> float:
        """R² of using lagged predictor values to predict target."""
        n = min(len(predictor), len(target))
        if n < lag * 3:
            return 0.0

        X = np.column_stack([predictor[lag - i - 1:n - i - 1] for i in range(lag)])
        Y = target[lag:]

        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            Y_hat = X @ beta
            ss_res = np.sum((Y - Y_hat) ** 2)
            ss_tot = np.sum((Y - np.mean(Y)) ** 2)
            if ss_tot < 1e-12:
                return 0.0
            return float(max(0.0, 1.0 - ss_res / ss_tot))
        except (np.linalg.LinAlgError, ValueError):
            return 0.0
