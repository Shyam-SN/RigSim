"""Sync strobe blink pattern generator and scene manipulator."""

import mujoco
import numpy as np


class SyncStrobeSimulator:
    """Simulates a synchronized optical strobe emitter with a pseudo-random binary sequence (PRBS)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        baud_rate_hz: float = 10.0,
        seed: int = 1337,
    ):
        self.model = model
        self.baud_rate_hz = baud_rate_hz
        self.symbol_period_s = 1.0 / baud_rate_hz
        self.rng = np.random.default_rng(seed)

        # Precompute pseudo-random bit sequence for up to 1000 seconds
        total_bits = int(1000 * baud_rate_hz)
        # Use balanced PRBS (approx 50% 1s and 0s)
        self.bits = self.rng.choice([0, 1], size=total_bits, p=[0.5, 0.5])

        # Find strobe material / geom in model
        self.strobe_mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "strobe_mat")
        self.strobe_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sync_strobe_geom")

    def get_strobe_state(self, t_sim_s: float) -> int:
        """Return 1 (ON) or 0 (OFF) for the current simulation instant."""
        idx = int(t_sim_s / self.symbol_period_s) % len(self.bits)
        return int(self.bits[idx])

    def update_model(self, t_sim_s: float) -> None:
        """Update MuJoCo material emission and rgba dynamically before rendering."""
        state = self.get_strobe_state(t_sim_s)

        if self.strobe_mat_id >= 0:
            if state == 1:
                # Flashing bright white with high emission
                self.model.mat_emission[self.strobe_mat_id] = 1.0
                self.model.mat_rgba[self.strobe_mat_id] = [1.0, 1.0, 1.0, 1.0]
            else:
                # Dim / OFF
                self.model.mat_emission[self.strobe_mat_id] = 0.0
                self.model.mat_rgba[self.strobe_mat_id] = [0.1, 0.1, 0.1, 1.0]
