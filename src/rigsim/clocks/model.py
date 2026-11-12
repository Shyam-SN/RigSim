"""Sensor clock models and transport latency simulation."""

import numpy as np

from rigsim.config.clock import ClockModel, TransportLatencyModel


class ClockEngine:
    """Simulates realistic crystal oscillator behavior.

    t_device = (1 + drift_ppm * 1e-6) * t_sim + offset_ns + N(0, jitter_std_ns) + step_offset
    """

    def __init__(self, model: ClockModel, seed: int = 42):
        self.clock_id = model.clock_id
        self.offset_ns = model.offset_ns
        self.drift_ppm = model.drift_ppm
        self.jitter_std_ns = model.jitter_std_ns
        self.step_events = sorted(model.step_events, key=lambda e: e.t_sim_ns)
        self.rng = np.random.default_rng(seed)

    def sim_to_device_time_ns(self, t_sim_ns: int) -> int:
        """Convert simulation ground-truth time (ns) to sensor device clock time (ns)."""
        # Linear frequency drift
        drift_factor = 1.0 + (self.drift_ppm * 1e-6)
        base_t = drift_factor * t_sim_ns + self.offset_ns

        # Add step discontinuities if any step event has occurred
        step_delta = 0
        for ev in self.step_events:
            if t_sim_ns >= ev.t_sim_ns:
                step_delta += ev.step_ns
        base_t += step_delta

        # Add Gaussian jitter
        if self.jitter_std_ns > 0.0:
            jitter = self.rng.normal(0.0, self.jitter_std_ns)
            base_t += jitter

        return int(np.round(base_t))


class TransportDelayEngine:
    """Simulates transport latency from sensor to host OS."""

    def __init__(self, model: TransportLatencyModel | None = None, seed: int = 42):
        self.model = model or TransportLatencyModel()
        self.rng = np.random.default_rng(seed)

    def compute_host_time_ns(self, device_time_ns: int) -> int:
        """Compute host arrival timestamp (log_time) from sensor publish time."""
        latency_ms = self.rng.normal(self.model.mean_ms, self.model.jitter_std_ms)
        latency_ms = max(0.5, latency_ms)  # Lower bound

        # Latency spike
        if self.rng.uniform(0.0, 1.0) < self.model.spike_prob:
            latency_ms *= self.model.spike_multiplier

        latency_ns = int(round(latency_ms * 1_000_000))
        return device_time_ns + latency_ns
