#!/usr/bin/env python3
"""
Full experiment: test whether memory signals encode weight/parameter dynamics.

Runs three experiments:
1. Healthy training — stable convergence, correlate memory with weights
2. Injected instability — spike LR mid-training, test detection lag
3. Landscape shift — move loss surface, test early warning capability

For each: compute cross-correlations, Granger causality, predictive R²,
and per-layer breakdown.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memtrace.weight_simulator import WeightDynamicsSimulator, NetworkConfig
from memtrace.correlator import SignalCorrelator
from memtrace.analyzer import SpectralAnalyzer, ChaosDetector


def run_experiment(
    name: str,
    sim: WeightDynamicsSimulator,
    steps: int,
    lr_schedule: str = "warmup_cosine",
) -> dict:
    """Run one experiment and return full analysis."""
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"{'='*60}")

    sim.run(steps=steps, lr_schedule=lr_schedule)
    ts = sim.get_timeseries()
    layer_names = list(sim.layers.keys())

    # correlation analysis
    correlator = SignalCorrelator(window_size=40)
    corr = correlator.full_analysis(ts, layer_names)

    # spectral analysis on memory
    mem_analyzer = SpectralAnalyzer(sampling_rate=1.0)
    mem_spectral = mem_analyzer.analyze(ts["memory_noisy"])
    chaos_det = ChaosDetector(window_size=40)
    chaos = chaos_det.analyze(ts["memory_noisy"], sampling_rate=1.0)

    # print results
    print(f"\n  --- Correlation Results ---")
    print(f"  Memory ↔ Grad norm:     r={corr.memory_vs_grad_norm.pearson_r:.3f}, lag={corr.memory_vs_grad_norm.lag_at_peak}")
    print(f"  Memory ↔ Loss:          r={corr.memory_vs_loss.pearson_r:.3f}, lag={corr.memory_vs_loss.lag_at_peak}")
    print(f"  Memory ↔ Param velocity:r={corr.memory_vs_param_velocity.pearson_r:.3f}, lag={corr.memory_vs_param_velocity.lag_at_peak}")
    print(f"  Memory ↔ Adam v-norm:   r={corr.memory_vs_adam_v_norm.pearson_r:.3f}, lag={corr.memory_vs_adam_v_norm.lag_at_peak}")
    print(f"\n  --- Spectral Correlations ---")
    print(f"  Mem entropy ↔ Grad var: r={corr.memory_entropy_vs_grad_variance.pearson_r:.3f}")
    print(f"  Mem entropy ↔ Weight Δ: r={corr.memory_entropy_vs_weight_norm_change.pearson_r:.3f}")
    print(f"\n  --- Predictive Power ---")
    print(f"  Memory → Loss R²:      {corr.memory_predictive_r2:.4f}")
    print(f"  Gradient → Loss R²:    {corr.gradient_predictive_r2:.4f}")
    print(f"  Memory adds info:       {corr.memory_adds_information}")
    print(f"  Memory leads loss:      {corr.memory_leads_loss} (by {corr.instability_detection_lag} steps)")
    print(f"  Memory leads gradients: {corr.memory_leads_gradients}")
    print(f"\n  --- Per-Layer ---")
    for ln, lc in corr.layer_correlations.items():
        print(f"    {ln}: r={lc.pearson_r:.3f}, MI={lc.mutual_info:.3f}, Granger F={lc.granger_f_stat:.2f}")
    print(f"\n  --- Memory Diagnostics ---")
    print(f"  Spectral entropy:  {mem_spectral.spectral_entropy:.3f}")
    print(f"  Instability score: {chaos.instability_score:.3f}")
    print(f"  Hurst exponent:    {chaos.hurst_exponent:.3f}")

    # build JSON output
    ds = 1  # no downsampling for per-step data
    result = {
        "name": name,
        "steps": steps,
        # time series (for plotting)
        "step": ts["step"].tolist(),
        "loss": ts["loss"].tolist(),
        "learning_rate": ts["learning_rate"].tolist(),
        "total_grad_norm": ts["total_grad_norm"].tolist(),
        "total_weight_norm": ts["total_weight_norm"].tolist(),
        "param_velocity": ts["param_velocity"].tolist(),
        "memory_noisy": ts["memory_noisy"].tolist(),
        "memory_forward": ts["memory_forward"].tolist(),
        "adam_m_norm": ts["adam_m_norm"].tolist(),
        "adam_v_norm": ts["adam_v_norm"].tolist(),
        "effective_lr": ts["effective_lr"].tolist(),
        # per-layer
        "layer_names": layer_names,
    }

    for ln in layer_names:
        result[f"weight_norm_{ln}"] = ts[f"weight_norm_{ln}"].tolist()
        result[f"grad_norm_{ln}"] = ts[f"grad_norm_{ln}"].tolist()
        result[f"cosine_sim_{ln}"] = ts[f"cosine_sim_{ln}"].tolist()
        result[f"grad_snr_{ln}"] = ts[f"grad_snr_{ln}"].tolist()

    # correlation results
    result["correlations"] = {
        "memory_vs_grad": {
            "pearson_r": corr.memory_vs_grad_norm.pearson_r,
            "spearman_rho": corr.memory_vs_grad_norm.spearman_rho,
            "lag": corr.memory_vs_grad_norm.lag_at_peak,
            "peak_xcorr": corr.memory_vs_grad_norm.peak_xcorr,
            "granger_f": corr.memory_vs_grad_norm.granger_f_stat,
            "mutual_info": corr.memory_vs_grad_norm.mutual_info,
        },
        "memory_vs_loss": {
            "pearson_r": corr.memory_vs_loss.pearson_r,
            "lag": corr.memory_vs_loss.lag_at_peak,
            "granger_f": corr.memory_vs_loss.granger_f_stat,
            "mutual_info": corr.memory_vs_loss.mutual_info,
        },
        "memory_vs_velocity": {
            "pearson_r": corr.memory_vs_param_velocity.pearson_r,
            "lag": corr.memory_vs_param_velocity.lag_at_peak,
            "mutual_info": corr.memory_vs_param_velocity.mutual_info,
        },
        "entropy_vs_grad_var": {
            "pearson_r": corr.memory_entropy_vs_grad_variance.pearson_r,
            "lag": corr.memory_entropy_vs_grad_variance.lag_at_peak,
            "mutual_info": corr.memory_entropy_vs_grad_variance.mutual_info,
        },
        "memory_predictive_r2": corr.memory_predictive_r2,
        "gradient_predictive_r2": corr.gradient_predictive_r2,
        "memory_adds_information": corr.memory_adds_information,
        "memory_leads_loss": corr.memory_leads_loss,
        "memory_leads_gradients": corr.memory_leads_gradients,
        "instability_detection_lag": corr.instability_detection_lag,
        "per_layer": {
            ln: {
                "pearson_r": lc.pearson_r,
                "mutual_info": lc.mutual_info,
                "granger_f": lc.granger_f_stat,
                "lag": lc.lag_at_peak,
            }
            for ln, lc in corr.layer_correlations.items()
        },
    }

    # memory diagnostics
    result["diagnostics"] = {
        "spectral_entropy": float(mem_spectral.spectral_entropy),
        "instability_score": float(chaos.instability_score),
        "hurst_exponent": float(chaos.hurst_exponent),
        "sample_entropy": float(chaos.sample_entropy),
    }

    return result


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    network = NetworkConfig.mlp(
        [512, 256, 128, 64, 32, 10],
        ["embed", "hidden1", "hidden2", "hidden3", "head"],
    )
    N = 400

    # =======================================================================
    # Experiment 1: Healthy training (stable convergence)
    # =======================================================================
    sim1 = WeightDynamicsSimulator(
        network=network,
        learning_rate=1e-3,
        landscape_condition=10.0,
        allocator_noise=0.02,
        seed=42,
    )
    exp1 = run_experiment("healthy_training", sim1, steps=N, lr_schedule="warmup_cosine")

    # =======================================================================
    # Experiment 2: LR spike at step 200 (injected instability)
    # =======================================================================
    sim2 = WeightDynamicsSimulator(
        network=network,
        learning_rate=1e-3,
        landscape_condition=10.0,
        allocator_noise=0.02,
        seed=42,
    )
    sim2.inject_event(200, "lr_spike", factor=15.0)
    sim2.inject_event(220, "lr_restore")  # restore to normal
    exp2 = run_experiment("lr_spike", sim2, steps=N, lr_schedule="warmup_cosine")

    # =======================================================================
    # Experiment 3: Landscape shift at step 150 (distribution shift)
    # =======================================================================
    sim3 = WeightDynamicsSimulator(
        network=network,
        learning_rate=1e-3,
        landscape_condition=20.0,  # harder landscape
        allocator_noise=0.02,
        seed=42,
    )
    sim3.inject_event(150, "landscape_shift", magnitude=5.0)
    exp3 = run_experiment("landscape_shift", sim3, steps=N, lr_schedule="warmup_cosine")

    # =======================================================================
    # Experiment 4: Gradient corruption (simulates data poison / NaN)
    # =======================================================================
    sim4 = WeightDynamicsSimulator(
        network=network,
        learning_rate=1e-3,
        landscape_condition=10.0,
        allocator_noise=0.02,
        seed=42,
    )
    sim4.inject_event(180, "grad_corrupt", scale=100.0)
    exp4 = run_experiment("grad_corruption", sim4, steps=N, lr_schedule="warmup_cosine")

    # save all
    all_results = {
        "healthy_training": exp1,
        "lr_spike": exp2,
        "landscape_shift": exp3,
        "grad_corruption": exp4,
    }
    outfile = output_dir / "weight_dynamics_results.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f)
    print(f"\n{'='*60}")
    print(f"  All results saved to {outfile}")
    print(f"  Total size: {outfile.stat().st_size / 1024:.0f} KB")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    main()
