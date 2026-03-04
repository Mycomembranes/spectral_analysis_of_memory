# memtrace-diagnostics

**Training optimization health diagnostics via memory signal spectral analysis.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)


---

## The Idea

During neural network training, VRAM/RAM allocation follows a mostly deterministic sawtooth pattern: forward pass → activations accumulate → backward → gradients freed → optimizer step. **When optimization is healthy, this signal is clean and periodic. When it isn't, it becomes chaotic — often *before* the loss curve shows problems.**

`memtrace-diagnostics` treats memory allocation as a 1D time-series signal and applies spectral analysis to detect instability:

| Regime | Memory Pattern | FFT Spectrum | Spectral Entropy |
|---|---|---|---|
| **Stable** | Clean periodic sawtooth | Sharp peaks at batch freq | Low (~2 bits) |
| **Transitional** | Growing variance, sporadic spikes | Broadening peaks | Medium (~3 bits) |
| **Chaotic** | Heavy-tailed noise, fragmentation | Broadband / flat | High (~7+ bits) |

## Quick Start

```bash
pip install memtrace-diagnostics
```

### Drop into any training loop

```python
from memtrace import TrainingMonitor

monitor = TrainingMonitor(backend="cuda")

for step, batch in enumerate(loader):
    monitor.step_start(step)
    loss = model(batch)
    loss.backward()
    optimizer.step()
    
    alert = monitor.step_end(step, loss=loss.item())
    if alert:
        print(f"⚠ Step {step}: {alert['message']}")
        # Optionally: reduce LR, increase gradient clipping, checkpoint
```

### Synthetic simulation & analysis

```python
from memtrace import SyntheticTrainingSimulator, SpectralAnalyzer, ChaosDetector

# Generate realistic memory traces
sim = SyntheticTrainingSimulator(steps=300, seed=42)
trace = sim.generate("chaotic")

# Spectral analysis
analyzer = SpectralAnalyzer(sampling_rate=10.0)
result = analyzer.analyze(trace.memory_mb)
print(f"Spectral entropy: {result.spectral_entropy:.3f}")
print(f"Dominant frequency: {result.dominant_freq:.3f} Hz")

# Chaos detection
detector = ChaosDetector(window_size=200)
metrics = detector.analyze(trace.memory_mb)
print(f"Instability score: {metrics.instability_score:.3f}")
print(f"Hurst exponent: {metrics.hurst_exponent:.3f}")
print(f"Unstable regions: {len(metrics.instability_regions)}")
```

## Architecture

```
memtrace/
├── collector.py     # MemoryCollector (real GPU/CPU) + SyntheticTrainingSimulator
├── analyzer.py      # SpectralAnalyzer (FFT/PSD) + ChaosDetector (entropy, Hurst, etc.)
└── monitor.py       # TrainingMonitor — drop-in callback for live alerting
```

### Key Metrics

| Metric | What it measures | Healthy range |
|---|---|---|
| **Spectral Entropy** | Disorder of frequency content (Shannon entropy of normalized PSD) | < 4.0 bits |
| **Instability Score** | Composite of entropy + CV + Hurst deviation (0–1 scale) | < 0.5 |
| **Hurst Exponent** | Memory in the time-series (0.5 = random walk, >0.5 = trending) | 0.4–0.6 |
| **Sample Entropy** | Regularity/predictability of the signal | < 1.0 |
| **Dominant Frequency** | Strongest periodic component (should match batch/epoch rate) | Matches batch freq |
| **90% Bandwidth** | Spectral spread — how concentrated vs. diffuse the energy is | Narrow |

### Simulation Regimes

The `SyntheticTrainingSimulator` generates physically-motivated traces:

- **Stable**: Low-variance Gaussian noise on a deterministic sawtooth. Models a well-converged optimizer with predictable compute patterns.
- **Transitional**: Progressively growing variance with intermittent exponential spikes. Models an optimizer entering a new loss-landscape region (gradient bursts, momentum adjusting).
- **Chaotic**: Heavy-tailed (Student-t) noise, correlated random-walk drift, Poisson fragmentation events. Models exploding gradients, optimizer state divergence, and allocator thrashing.

## Validation Strategy

To confirm this heuristic provides *actionable signal beyond what's trivially available*:

1. **Instrument** a training run with both `torch.cuda.memory_allocated()` AND gradient norms logged per step
2. **Compute** rolling spectral entropy on memory vs. rolling variance of gradient norms
3. **Test** whether memory entropy *leads* gradient norm variance at instability onset
4. **Deliberately induce** instability (spike LR, corrupt batch) and measure detection latency

```python
# Correlation test scaffold
from memtrace import MemoryCollector, SpectralAnalyzer
import torch

collector = MemoryCollector(backend="cuda")
analyzer = SpectralAnalyzer()
grad_norms = []

for step, batch in enumerate(loader):
    collector.record(step, phase="forward")
    loss = model(batch)
    loss.backward()
    
    # Capture gradient norm
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    grad_norms.append(total_norm.item())
    
    optimizer.step()
    collector.record(step, phase="optimizer")

# Post-hoc analysis
mem = collector.trace.memory_mb
mem_spectral = analyzer.analyze(mem)
print(f"Memory spectral entropy: {mem_spectral.spectral_entropy:.3f}")
print(f"Grad norm CV: {np.std(grad_norms) / np.mean(grad_norms):.3f}")
```

## Caveats

- **Allocator noise**: PyTorch's caching allocator and CUDA memory pooling add artifacts. GPU VRAM (`torch.cuda.memory_allocated()`) is cleaner than host RAM.
- **Confounders**: Data loader prefetching, GC pauses, mixed-precision ops, multi-GPU sharding can inject non-optimization noise. Calibrate per setup.
- **Scale**: Most diagnostic at mid-to-large model scale where activations/optimizer states dominate fluctuations.
- **Lossy projection**: Memory is a summed effect across all parameters — layer-wise profiling (`torch.profiler`) can provide finer resolution.

## Optional Dependencies

```bash
pip install memtrace-diagnostics[gpu]     # + torch
pip install memtrace-diagnostics[viz]     # + matplotlib
pip install memtrace-diagnostics[all]     # everything
```

## License

