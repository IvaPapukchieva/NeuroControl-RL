"""
Storage of all the parameters we need, decision interval, learning rate, temperature etc
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    simulation_length: float = 15000.0  # ms
    step_size: float = 0.001            # ms, integration timestep
    sampling_interval: float = 1.0      # ms
    r: float = 0.00035                  # Jirsa(2014) default
    noise_seed: int = 42
#TODO: Do something about noise_seed

    # State order: x1, y1, z, x2, y2, g
    initial_state: tuple = (
        0.0,   # x1
        -5.0,  # y1
        3.0,   # z
        0.0,   # x2
        0.0,   # y2
        0.0,   # g
    )   

    # sqrt(Variance) of additive noise for each state variable in the Epileptor model
    # Order: x1, y1, z, x2, y2, g
    noise_nsig: tuple = (
        0.0,    # x1
        0.0,    # y1
        0.0,    # z
        1e-3,   # x2
        1e-3,   # y2
        0.0,    # g
    )

    @property
    def fs(self) -> float:
        """Output sampling rate in Hz."""
        return 1000.0 / self.sampling_interval

    @property
    def noise_array(self) -> np.ndarray:
        """Noise parameter array shaped for the TVB state."""
        return np.array(self.noise_nsig).reshape(6, 1, 1)

    @property
    def initial_state_array(self) -> np.ndarray:
        """Initial state shaped for TVB: (variables, nodes, modes)."""
        return np.array(self.initial_state, dtype=float).reshape(6, 1, 1)

CONFIGS = {
    "default": SimulationConfig(),

    "high_resolution": SimulationConfig(
        step_size=0.001,
        sampling_interval=1.0,
    ),

    "fast": SimulationConfig(
        step_size=0.01,
        sampling_interval=1.0,
    ),

    "jirsa_2014": SimulationConfig(
        r=0.00035,
        noise_seed=42,
    ),

    "short_test": SimulationConfig(
        simulation_length=1000.0,
        step_size=0.01,
    ),
}