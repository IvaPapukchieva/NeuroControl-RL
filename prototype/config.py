"""
Storage of all the parameters we need, decision interval, learning rate, temperature etc
"""
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    simulation_length: float = 15000.0  # ms
    time_step: float = 0.001            # ms, integration timestep
    sampling_interval: float = 1.0      # ms, output interval
    fs: float = 1000.0                  # Hz, output sampling rate

    r: float = 0.00125
    noise_seed: int = 42
