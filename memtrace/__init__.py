"""
memtrace-diagnostics: Training optimization health diagnostics via memory signal analysis.

Uses RAM/VRAM allocation patterns as a proxy trace for optimization dynamics,
applying spectral analysis to detect training instabilities before they manifest in loss.
"""

__version__ = "0.1.0"
__author__ = "Mukshud Ahamed"

from memtrace.collector import MemoryCollector, SyntheticTrainingSimulator
from memtrace.analyzer import SpectralAnalyzer, ChaosDetector
from memtrace.monitor import TrainingMonitor
from memtrace.weight_simulator import WeightDynamicsSimulator, NetworkConfig
from memtrace.correlator import SignalCorrelator

__all__ = [
    "MemoryCollector",
    "SyntheticTrainingSimulator",
    "SpectralAnalyzer",
    "ChaosDetector",
    "TrainingMonitor",
    "WeightDynamicsSimulator",
    "NetworkConfig",
    "SignalCorrelator",
]
