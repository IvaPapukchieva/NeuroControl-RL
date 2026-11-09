"""
Storage of all the parameters we need, decision interval, learning rate, temperature etc
"""
import numpy as np
from dataclasses import dataclass, field
from tvb.simulator.models.epileptor import Epileptor
from prototype.epileptor import ChizhovEpileptor2

@dataclass
class SimulationConfig:
    simulation_length: float = 5000.0  # ms
    step_size: float = 0.001            # == 1ms, integration timestep
    sampling_interval: float = 1.0      # ms

    model_class: type = Epileptor
    model_params: dict = field(default_factory=dict)
    noise_seed: int = 42

    # State order: x1, y1, z, x2, y2, g
    initial_state: tuple = (0.0,-5.0,3.0,0.0,0.0,0.0)   

    # sqrt(Variance) of additive noise for each state variable in the Epileptor model
    # Order: x1, y1, z, x2, y2, g
    noise_nsig: tuple = (0.0,0.0,0.0,1e-3,1e-3,0.0)

    @property
    def fs(self) -> float:
        """Output sampling rate in Hz."""
        return 1000.0 / self.sampling_interval

    def state_array(self, nvar: int) -> np.ndarray:
        """Convert initial conditions to TVB state shape."""
        if len(self.initial_state) != nvar:
            raise ValueError(
                f"Expected {nvar} initial values, "
                f"got {len(self.initial_state)}"
            )

        return np.asarray(
            self.initial_state, dtype=float
        ).reshape(nvar, 1, 1)

    def noise_array(self, nvar: int) -> np.ndarray:
        """Convert noise intensities to TVB state shape."""
        if len(self.noise_nsig) != nvar:
            raise ValueError(
                f"Expected {nvar} noise values, "
                f"got {len(self.noise_nsig)}"
            )

        return np.asarray(
            self.noise_nsig, dtype=float
        ).reshape(nvar, 1, 1)

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
        model_params={"r": 0.00035},
        step_size=0.01,
        noise_seed=42,
    ),

    "nagaraj_2017": SimulationConfig(
        model_params={"r": 0.00125},  # 800s; 400s => r=0.0025
        noise_seed=42,
        step_size=0.001,
        initial_state=(0.0, -5.0, 3.0, 0.0, 0.0, 0.0),
        noise_nsig=(0.0, 0.0, 0.0, 1e-3, 1e-3, 0.0),
    ),

    "jirsa_2014_p31_case1": SimulationConfig(
        step_size=0.01,
        model_params={"r": 0.00035},
        noise_seed=42,
        initial_state=(0.022, 0.91, 3.84, -1.11, 0.73, 0.0),
        noise_nsig=(0.0003125, 0.0003125, 0.0, 0.005, 0.005, 0.0),
    ),

    "no_noise": SimulationConfig(
        step_size=0.01,
        model_params={"r": 0.00035},
        noise_seed=42,
        initial_state=(0.022, 0.91, 3.84, -1.11, 0.73, 0.0),
        noise_nsig=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),

    "chizhov_2018": SimulationConfig(
        model_class=ChizhovEpileptor2,
        model_params={"tau_K": 100_000.0},
        simulation_length=600_000.0,
        step_size=0.1,
        sampling_interval=1.0,
        initial_state=(3.0, 10.0, 0.0, 1.0),
        noise_nsig=(0.0, 0.0, 31.25, 0.0),
    ),
}