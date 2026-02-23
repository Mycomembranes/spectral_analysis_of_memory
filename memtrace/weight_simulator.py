"""
Synthetic training engine with full parameter-space observability.

Simulates a multi-layer neural network with real linear-algebra weight dynamics,
Adam optimizer state, and physically-derived memory allocation. This provides
ground truth for testing whether memory spectral features can recover information
about what weights and parameters are actually doing.

The key insight: memory allocation at each phase is a *deterministic function*
of the current parameter state plus allocator noise. If spectral structure in
the memory signal correlates with spectral structure in weight-space dynamics,
then memory IS encoding optimization health — not just hardware artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np


# ---------------------------------------------------------------------------
# Network architecture definition
# ---------------------------------------------------------------------------

@dataclass
class LayerConfig:
    """Single layer specification."""
    name: str
    input_dim: int
    output_dim: int
    activation: str = "relu"  # relu, tanh, linear


@dataclass
class NetworkConfig:
    """Full network architecture."""
    layers: list[LayerConfig]
    dtype_bytes: int = 4  # float32

    @property
    def total_params(self) -> int:
        return sum(l.input_dim * l.output_dim + l.output_dim for l in self.layers)

    @property
    def param_bytes(self) -> int:
        return self.total_params * self.dtype_bytes

    @classmethod
    def mlp(cls, dims: list[int], names: Optional[list[str]] = None) -> "NetworkConfig":
        """Create a standard MLP architecture."""
        layers = []
        for i in range(len(dims) - 1):
            name = names[i] if names else f"layer_{i}"
            act = "relu" if i < len(dims) - 2 else "linear"
            layers.append(LayerConfig(name, dims[i], dims[i + 1], act))
        return cls(layers=layers)


# ---------------------------------------------------------------------------
# Layer state — weights, gradients, optimizer moments
# ---------------------------------------------------------------------------

@dataclass
class LayerState:
    """Full observable state of a single layer."""
    config: LayerConfig
    weights: np.ndarray          # (out, in) weight matrix
    bias: np.ndarray             # (out,) bias vector
    grad_w: np.ndarray           # weight gradients
    grad_b: np.ndarray           # bias gradients
    adam_m_w: np.ndarray         # first moment (mean of gradients)
    adam_v_w: np.ndarray         # second moment (variance of gradients)
    adam_m_b: np.ndarray
    adam_v_b: np.ndarray
    activations: Optional[np.ndarray] = None  # cached forward-pass activations

    @property
    def weight_norm(self) -> float:
        return float(np.linalg.norm(self.weights))

    @property
    def grad_norm(self) -> float:
        return float(np.linalg.norm(self.grad_w))

    @property
    def update_magnitude(self) -> float:
        """Magnitude of the Adam update (before applying)."""
        eps = 1e-8
        update = self.adam_m_w / (np.sqrt(self.adam_v_w) + eps)
        return float(np.linalg.norm(update))

    @property
    def grad_snr(self) -> float:
        """Gradient signal-to-noise ratio."""
        mean = np.mean(np.abs(self.grad_w))
        std = np.std(self.grad_w)
        return float(mean / (std + 1e-12))

    @property
    def weight_bytes(self) -> int:
        return self.weights.nbytes + self.bias.nbytes

    @property
    def grad_bytes(self) -> int:
        return self.grad_w.nbytes + self.grad_b.nbytes

    @property
    def optimizer_bytes(self) -> int:
        return (self.adam_m_w.nbytes + self.adam_v_w.nbytes +
                self.adam_m_b.nbytes + self.adam_v_b.nbytes)

    @property
    def activation_bytes(self) -> int:
        return self.activations.nbytes if self.activations is not None else 0

    def memory_footprint_dynamic(self, phase: str) -> int:
        """
        Physically-motivated memory that depends on actual parameter state.

        In real training, memory allocation is coupled to dynamics through:
        1. Activation magnitudes → caching allocator headroom & mixed-precision upcasts
        2. Gradient magnitudes → intermediate autograd tensor sizes
        3. Optimizer moment magnitudes → precision guard bands
        4. Fragmentation → increases when allocation patterns change (instability)

        Coupling strengths calibrated to approximate real PyTorch caching allocator
        behavior on GPU: activations can 2-3x memory with large magnitudes,
        gradients add 50-200% overhead during instability.
        """
        base = self.weight_bytes + self.optimizer_bytes

        # (1) Activation-dependent: RMS > 1.0 triggers increasing headroom
        # Real: large activations cause allocator block splits and fp16→fp32 upcasts
        act_scale = 1.0
        if self.activations is not None:
            act_rms = float(np.sqrt(np.mean(self.activations ** 2)))
            # nonlinear: small activations ~ 1x, large activations up to 3x
            act_scale = 1.0 + 0.8 * np.tanh(max(0, act_rms - 0.5))

        # (2) Gradient-dependent: large grads create more autograd intermediates
        # Real: gradient accumulation, mixed-precision scaling, clip buffers
        grad_rms = float(np.sqrt(np.mean(self.grad_w ** 2))) if self.grad_w.size > 0 else 0.0
        grad_scale = 1.0 + 1.0 * np.tanh(grad_rms / 2.0)  # saturates at 2x

        # (3) Optimizer moment pressure: large v → numerical precision buffers
        v_rms = float(np.sqrt(np.mean(self.adam_v_w ** 2))) if self.adam_v_w.size > 0 else 0.0
        opt_scale = 1.0 + 0.3 * np.tanh(v_rms)

        if phase == "forward":
            return int(base + self.activation_bytes * act_scale)
        elif phase == "backward":
            return int(base + self.activation_bytes * act_scale
                       + self.grad_bytes * grad_scale)
        elif phase == "optimizer":
            return int(base * opt_scale + self.grad_bytes * grad_scale)
        return base


# ---------------------------------------------------------------------------
# Per-step snapshot of the full network
# ---------------------------------------------------------------------------

@dataclass
class TrainingSnapshot:
    """Complete observable state at one training step."""
    step: int
    loss: float
    learning_rate: float

    # per-layer metrics (indexed by layer name)
    weight_norms: dict[str, float]
    grad_norms: dict[str, float]
    update_magnitudes: dict[str, float]
    grad_snrs: dict[str, float]
    weight_cosine_sim: dict[str, float]  # similarity to previous step

    # global aggregates
    total_grad_norm: float
    total_weight_norm: float
    param_velocity: float  # ||θ_t - θ_{t-1}||

    # physically-derived memory
    memory_forward_bytes: int
    memory_backward_bytes: int
    memory_optimizer_bytes: int
    memory_with_noise_bytes: int  # includes allocator artifacts

    # optimizer health
    adam_m_norm: float   # first moment global norm
    adam_v_norm: float   # second moment global norm
    effective_lr: float  # actual step size after Adam scaling


# ---------------------------------------------------------------------------
# Loss landscape
# ---------------------------------------------------------------------------

class SyntheticLossLandscape:
    """
    Parameterized loss function with controllable difficulty.

    Generates a loss landscape as a sum of quadratic bowls plus
    optional saddle points, narrow valleys, and noise — providing
    a rich optimization surface where different regimes naturally
    emerge based on learning rate and landscape parameters.
    """

    def __init__(
        self,
        dim: int,
        n_bowls: int = 3,
        n_saddles: int = 1,
        noise_scale: float = 0.01,
        condition_number: float = 10.0,
        seed: int = 42,
    ):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

        # generate bowl centers and curvatures
        self.bowl_centers = [self.rng.standard_normal(dim) * 2 for _ in range(n_bowls)]
        self.bowl_weights = self.rng.dirichlet(np.ones(n_bowls))

        # anisotropic curvature (condition number controls difficulty)
        eigenvalues = np.logspace(0, np.log10(condition_number), min(dim, 50))
        if dim > 50:
            eigenvalues = np.concatenate([eigenvalues, np.ones(dim - 50)])
        self.rng.shuffle(eigenvalues)
        Q = self._random_orthogonal(dim)
        self.hessian = Q @ np.diag(eigenvalues[:dim]) @ Q.T

        # saddle point directions
        self.saddle_dirs = [self.rng.standard_normal(dim) for _ in range(n_saddles)]
        self.saddle_strength = 0.3

        self.noise_scale = noise_scale

    def _random_orthogonal(self, n: int) -> np.ndarray:
        H = self.rng.standard_normal((n, n))
        Q, _ = np.linalg.qr(H)
        return Q

    def __call__(self, params_flat: np.ndarray, add_noise: bool = True) -> float:
        """Compute loss at given parameter vector."""
        loss = 0.0
        for center, weight in zip(self.bowl_centers, self.bowl_weights):
            diff = params_flat[:len(center)] - center
            loss += weight * 0.5 * diff @ self.hessian[:len(diff), :len(diff)] @ diff

        # saddle contribution
        for d in self.saddle_dirs:
            proj = np.dot(params_flat[:len(d)], d / (np.linalg.norm(d) + 1e-8))
            loss -= self.saddle_strength * proj ** 2

        if add_noise:
            loss += self.rng.normal(0, self.noise_scale * (1 + abs(loss) * 0.01))

        return float(max(loss, 0.0))

    def gradient(self, params_flat: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Analytical gradient with optional stochastic noise."""
        grad = np.zeros_like(params_flat)
        for center, weight in zip(self.bowl_centers, self.bowl_weights):
            n = len(center)
            diff = params_flat[:n] - center
            grad[:n] += weight * self.hessian[:n, :n] @ diff

        for d in self.saddle_dirs:
            n = len(d)
            d_norm = d / (np.linalg.norm(d) + 1e-8)
            proj = np.dot(params_flat[:n], d_norm)
            grad[:n] -= 2 * self.saddle_strength * proj * d_norm

        if add_noise:
            noise_mag = self.noise_scale * (1 + np.linalg.norm(grad) * 0.05)
            grad += self.rng.normal(0, noise_mag, grad.shape)

        return grad


# ---------------------------------------------------------------------------
# Training engine
# ---------------------------------------------------------------------------

class WeightDynamicsSimulator:
    """
    Full synthetic training simulation with ground-truth observability.

    Runs Adam optimization on a synthetic loss landscape with a multi-layer
    network. Memory allocation is derived *physically* from actual tensor
    sizes at each phase, plus configurable allocator noise.

    This allows direct correlation testing between memory signal features
    and weight-space dynamics.

    Parameters
    ----------
    network : NetworkConfig
        Network architecture.
    batch_size : int
        Simulated batch size (affects activation memory).
    learning_rate : float
        Initial learning rate.
    beta1, beta2 : float
        Adam momentum parameters.
    weight_decay : float
        L2 regularization.
    allocator_noise : float
        Fraction of total memory used as allocator noise std.
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        network: Optional[NetworkConfig] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 1e-4,
        allocator_noise: float = 0.02,
        landscape_condition: float = 10.0,
        seed: int = 42,
    ):
        if network is None:
            network = NetworkConfig.mlp(
                [512, 256, 128, 64, 32, 10],
                ["embed", "hidden1", "hidden2", "hidden3", "head"]
            )

        self.net = network
        self.batch_size = batch_size
        self.lr = learning_rate
        self.lr_init = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.wd = weight_decay
        self.alloc_noise = allocator_noise
        self.rng = np.random.default_rng(seed)

        # initialize layers
        self.layers: dict[str, LayerState] = {}
        for lc in network.layers:
            scale = np.sqrt(2.0 / lc.input_dim)  # He initialization
            self.layers[lc.name] = LayerState(
                config=lc,
                weights=self.rng.normal(0, scale, (lc.output_dim, lc.input_dim)),
                bias=np.zeros(lc.output_dim),
                grad_w=np.zeros((lc.output_dim, lc.input_dim)),
                grad_b=np.zeros(lc.output_dim),
                adam_m_w=np.zeros((lc.output_dim, lc.input_dim)),
                adam_v_w=np.zeros((lc.output_dim, lc.input_dim)),
                adam_m_b=np.zeros(lc.output_dim),
                adam_v_b=np.zeros(lc.output_dim),
            )

        # flatten params for loss landscape
        total_dim = min(network.total_params, 200)  # cap landscape dim
        self.landscape = SyntheticLossLandscape(
            dim=total_dim,
            condition_number=landscape_condition,
            seed=seed,
        )

        # history
        self.snapshots: list[TrainingSnapshot] = []
        self._prev_params_flat: Optional[np.ndarray] = None
        self._prev_weights: dict[str, np.ndarray] = {}
        self._step = 0

        # event injection queue
        self._events: list[dict] = []
        self._lr_multiplier: float = 1.0

    def _flatten_params(self) -> np.ndarray:
        """Flatten first N params for landscape evaluation."""
        parts = []
        for layer in self.layers.values():
            parts.extend([layer.weights.ravel(), layer.bias.ravel()])
        flat = np.concatenate(parts)
        return flat[:self.landscape.dim]

    def _apply_activation(self, x: np.ndarray, act: str) -> np.ndarray:
        if act == "relu":
            return np.maximum(x, 0)
        elif act == "tanh":
            return np.tanh(x)
        return x

    # -- event injection ----------------------------------------------------

    def inject_event(self, step: int, event_type: str, **kwargs):
        """
        Schedule an instability event at a given step.

        event_type:
            "lr_spike"     — multiply LR by factor (default 10x)
            "lr_restore"   — reset LR multiplier to 1.0
            "grad_corrupt" — add explosive noise to gradients
            "weight_reset" — randomly reinitialize a fraction of weights
            "landscape_shift" — shift loss landscape centers
            "momentum_clash" — set Adam betas to adversarial values
        """
        self._events.append({"step": step, "type": event_type, **kwargs})

    def _process_events(self, step: int):
        """Apply any scheduled events for this step."""
        for ev in self._events:
            if ev["step"] != step:
                continue
            if ev["type"] == "lr_spike":
                self._lr_multiplier = ev.get("factor", 10.0)
            elif ev["type"] == "weight_reset":
                frac = ev.get("fraction", 0.3)
                for layer in self.layers.values():
                    mask = self.rng.random(layer.weights.shape) < frac
                    scale = np.sqrt(2.0 / layer.config.input_dim)
                    layer.weights[mask] = self.rng.normal(0, scale, mask.sum())
            elif ev["type"] == "landscape_shift":
                mag = ev.get("magnitude", 3.0)
                for i in range(len(self.landscape.bowl_centers)):
                    self.landscape.bowl_centers[i] += self.rng.normal(0, mag, self.landscape.dim)
            elif ev["type"] == "momentum_clash":
                self.beta1 = ev.get("beta1", 0.1)
                self.beta2 = ev.get("beta2", 0.5)
            elif ev["type"] == "lr_restore":
                self._lr_multiplier = 1.0

    def _process_post_backward_events(self, step: int):
        """Apply gradient-modifying events AFTER backward pass."""
        for ev in self._events:
            if ev["step"] != step:
                continue
            if ev["type"] == "grad_corrupt":
                scale = ev.get("scale", 50.0)
                for layer in self.layers.values():
                    layer.grad_w += self.rng.normal(0, scale, layer.grad_w.shape)
                    layer.grad_b += self.rng.normal(0, scale, layer.grad_b.shape)

    # -- core training step -------------------------------------------------

    def step(self, lr_override: Optional[float] = None) -> TrainingSnapshot:
        """Execute one full training step and return observable snapshot."""
        self._step += 1
        step = self._step

        # process events
        self._process_events(step)
        lr = lr_override if lr_override is not None else self.lr
        lr *= self._lr_multiplier  # events multiply on top of schedule

        # save previous weights for cosine similarity
        prev_weights = {
            name: layer.weights.copy()
            for name, layer in self.layers.items()
        }

        # --- FORWARD PASS ---
        # Generate activations through actual weight matrix operations
        # so activation magnitudes are coupled to weight state
        x = self.rng.standard_normal(
            (self.batch_size, self.net.layers[0].input_dim)
        ).astype(np.float32)

        for name, layer in self.layers.items():
            # actual matmul: activations = x @ W^T + b
            x = x[:, :layer.config.input_dim] @ layer.weights.T + layer.bias
            x = self._apply_activation(x, layer.config.activation)
            layer.activations = x.copy()  # cached for backward

        mem_forward = sum(l.memory_footprint_dynamic("forward") for l in self.layers.values())

        # compute loss on flattened params
        params_flat = self._flatten_params()
        loss = self.landscape(params_flat)

        # --- BACKWARD PASS ---
        raw_grad = self.landscape.gradient(params_flat)

        # distribute gradient across layers (proportional to param count)
        offset = 0
        for name, layer in self.layers.items():
            n_w = layer.weights.size
            n_b = layer.bias.size

            # base gradient from landscape
            if offset + n_w <= len(raw_grad):
                grad_chunk_w = raw_grad[offset:offset + n_w].reshape(layer.weights.shape)
            else:
                grad_chunk_w = self.rng.normal(0, 0.01, layer.weights.shape)

            # add per-layer gradient noise (simulates mini-batch stochasticity)
            grad_noise = self.rng.normal(0, 0.01 * (1 + np.linalg.norm(grad_chunk_w)), layer.weights.shape)
            layer.grad_w = grad_chunk_w + grad_noise

            if offset + n_w + n_b <= len(raw_grad):
                layer.grad_b = raw_grad[offset + n_w:offset + n_w + n_b]
            else:
                layer.grad_b = self.rng.normal(0, 0.01, layer.bias.shape)

            offset += n_w + n_b

            # weight decay
            layer.grad_w += self.wd * layer.weights

        # apply gradient-modifying events (corruption, etc.)
        self._process_post_backward_events(step)

        mem_backward = sum(l.memory_footprint_dynamic("backward") for l in self.layers.values())

        # Capture activation statistics before optimizer frees them
        _total_act_rms = 0.0
        for layer in self.layers.values():
            if layer.activations is not None:
                _total_act_rms += float(np.sqrt(np.mean(layer.activations ** 2)))

        # --- OPTIMIZER STEP (Adam) ---
        eps = 1e-8
        bc1 = 1 - self.beta1 ** step
        bc2 = 1 - self.beta2 ** step

        for name, layer in self.layers.items():
            # update moments
            layer.adam_m_w = self.beta1 * layer.adam_m_w + (1 - self.beta1) * layer.grad_w
            layer.adam_v_w = self.beta2 * layer.adam_v_w + (1 - self.beta2) * layer.grad_w ** 2
            layer.adam_m_b = self.beta1 * layer.adam_m_b + (1 - self.beta1) * layer.grad_b
            layer.adam_v_b = self.beta2 * layer.adam_v_b + (1 - self.beta2) * layer.grad_b ** 2

            # bias-corrected estimates
            m_hat_w = layer.adam_m_w / bc1
            v_hat_w = layer.adam_v_w / bc2
            m_hat_b = layer.adam_m_b / bc1
            v_hat_b = layer.adam_v_b / bc2

            # update weights
            layer.weights -= lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
            layer.bias -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

            # free activations (simulates real training)
            layer.activations = None

        mem_optimizer = sum(l.memory_footprint_dynamic("optimizer") for l in self.layers.values())

        # --- ALLOCATOR NOISE (dynamics-correlated) ---
        # Real allocator noise is NOT independent of optimization state:
        # - Large gradient variance → irregular allocation patterns → fragmentation
        # - Changing activation magnitudes → allocator pool misses → new block allocations
        base_mem = max(mem_forward, mem_backward, mem_optimizer)

        # baseline independent noise (small)
        alloc_noise_bytes = int(abs(self.rng.normal(0, self.alloc_noise * base_mem * 0.5)))

        # dynamics-correlated fragmentation
        total_grad = sum(l.grad_norm for l in self.layers.values())
        # probability of fragmentation spike scales with grad magnitude
        frag_prob = min(0.25, 0.02 + 0.03 * total_grad)
        frag_scale = 1.0 + 1.5 * min(total_grad, 20.0)
        if self.rng.random() < frag_prob:
            alloc_noise_bytes += int(self.rng.exponential(
                self.alloc_noise * base_mem * frag_scale
            ))

        # activation-correlated overhead: when activations are large across
        # layers, the caching allocator has more active blocks
        alloc_noise_bytes += int(_total_act_rms * self.alloc_noise * base_mem * 0.1)

        mem_noisy = base_mem + alloc_noise_bytes

        # --- COMPUTE METRICS ---
        weight_norms = {n: l.weight_norm for n, l in self.layers.items()}
        grad_norms = {n: l.grad_norm for n, l in self.layers.items()}
        update_mags = {n: l.update_magnitude for n, l in self.layers.items()}
        grad_snrs = {n: l.grad_snr for n, l in self.layers.items()}

        # cosine similarity to previous step
        cosine_sims = {}
        for name, layer in self.layers.items():
            if name in prev_weights:
                old = prev_weights[name].ravel()
                new = layer.weights.ravel()
                dot = np.dot(old, new)
                norms = np.linalg.norm(old) * np.linalg.norm(new)
                cosine_sims[name] = float(dot / (norms + 1e-12))
            else:
                cosine_sims[name] = 1.0

        # parameter velocity
        new_flat = self._flatten_params()
        if self._prev_params_flat is not None:
            param_velocity = float(np.linalg.norm(new_flat - self._prev_params_flat))
        else:
            param_velocity = 0.0
        self._prev_params_flat = new_flat.copy()

        # Adam state norms
        adam_m_norm = float(np.sqrt(sum(
            np.sum(l.adam_m_w ** 2) + np.sum(l.adam_m_b ** 2)
            for l in self.layers.values()
        )))
        adam_v_norm = float(np.sqrt(sum(
            np.sum(l.adam_v_w ** 2) + np.sum(l.adam_v_b ** 2)
            for l in self.layers.values()
        )))

        # effective learning rate (Adam's actual step size)
        total_update = sum(l.update_magnitude for l in self.layers.values())
        total_grad = sum(l.grad_norm for l in self.layers.values())
        effective_lr = lr * total_update / (total_grad + 1e-12)

        snapshot = TrainingSnapshot(
            step=step,
            loss=loss,
            learning_rate=lr,
            weight_norms=weight_norms,
            grad_norms=grad_norms,
            update_magnitudes=update_mags,
            grad_snrs=grad_snrs,
            weight_cosine_sim=cosine_sims,
            total_grad_norm=float(np.sqrt(sum(v ** 2 for v in grad_norms.values()))),
            total_weight_norm=float(np.sqrt(sum(v ** 2 for v in weight_norms.values()))),
            param_velocity=param_velocity,
            memory_forward_bytes=mem_forward,
            memory_backward_bytes=mem_backward,
            memory_optimizer_bytes=mem_optimizer,
            memory_with_noise_bytes=mem_noisy,
            adam_m_norm=adam_m_norm,
            adam_v_norm=adam_v_norm,
            effective_lr=effective_lr,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def run(
        self,
        steps: int = 500,
        lr_schedule: Optional[str] = None,
        warmup_steps: int = 50,
    ) -> list[TrainingSnapshot]:
        """
        Run full training simulation.

        lr_schedule: None, "cosine", "step_decay", "warmup_cosine"
        """
        for s in range(steps):
            lr = self.lr_init
            if lr_schedule == "cosine":
                lr = self.lr_init * 0.5 * (1 + np.cos(np.pi * s / steps))
            elif lr_schedule == "step_decay":
                lr = self.lr_init * (0.1 ** (s // (steps // 3)))
            elif lr_schedule == "warmup_cosine":
                if s < warmup_steps:
                    lr = self.lr_init * s / warmup_steps
                else:
                    progress = (s - warmup_steps) / (steps - warmup_steps)
                    lr = self.lr_init * 0.5 * (1 + np.cos(np.pi * progress))
            self.step(lr_override=lr)
        return self.snapshots

    def get_timeseries(self) -> dict[str, np.ndarray]:
        """Extract all time series as numpy arrays for analysis."""
        n = len(self.snapshots)
        layer_names = list(self.layers.keys())

        ts = {
            "step": np.array([s.step for s in self.snapshots]),
            "loss": np.array([s.loss for s in self.snapshots]),
            "learning_rate": np.array([s.learning_rate for s in self.snapshots]),
            "total_grad_norm": np.array([s.total_grad_norm for s in self.snapshots]),
            "total_weight_norm": np.array([s.total_weight_norm for s in self.snapshots]),
            "param_velocity": np.array([s.param_velocity for s in self.snapshots]),
            "memory_forward": np.array([s.memory_forward_bytes for s in self.snapshots]) / 1e6,
            "memory_backward": np.array([s.memory_backward_bytes for s in self.snapshots]) / 1e6,
            "memory_optimizer": np.array([s.memory_optimizer_bytes for s in self.snapshots]) / 1e6,
            "memory_noisy": np.array([s.memory_with_noise_bytes for s in self.snapshots]) / 1e6,
            "adam_m_norm": np.array([s.adam_m_norm for s in self.snapshots]),
            "adam_v_norm": np.array([s.adam_v_norm for s in self.snapshots]),
            "effective_lr": np.array([s.effective_lr for s in self.snapshots]),
        }

        # per-layer series
        for name in layer_names:
            ts[f"weight_norm_{name}"] = np.array([s.weight_norms[name] for s in self.snapshots])
            ts[f"grad_norm_{name}"] = np.array([s.grad_norms[name] for s in self.snapshots])
            ts[f"update_mag_{name}"] = np.array([s.update_magnitudes[name] for s in self.snapshots])
            ts[f"cosine_sim_{name}"] = np.array([s.weight_cosine_sim[name] for s in self.snapshots])
            ts[f"grad_snr_{name}"] = np.array([s.grad_snrs[name] for s in self.snapshots])

        return ts
