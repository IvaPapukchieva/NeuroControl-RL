import numpy as np

from tvb.simulator.models.epileptor import Epileptor
from tvb.simulator.integrators import EulerStochastic
from tvb.simulator.noise import Additive

from prototype.config import SimulationConfig
from prototype.epileptor import NagarajEpileptor

class Simulator:
    def __init__(self, config: SimulationConfig):
        self.config = config

        self.total_steps = int(
            config.simulation_length / config.step_size
        )

        self.model = Epileptor()
        self.model.configure()
        self.model.r = np.array([config.r])

        self.integrator = EulerStochastic(
            dt=config.step_size,
            noise=Additive(
                nsig=config.noise_array,
                noise_seed=config.noise_seed,
            ),
        )
        self.integrator.configure()
        self.integrator.noise.dt = self.integrator.dt

        # State shape: (variables, nodes, modes)
        self.state = config.initial_state_array.copy()

        # These 3 are not necessary for a single node simulation, but are included for completeness
        self.coupling = np.zeros((1, 1, 1))
        self.local_coupling = 0.0
        self.stimulus = np.zeros_like(self.state)

    def next_state(self, new_stimulus=None):
        stimulus = (
            self.stimulus
            if new_stimulus is None
            else new_stimulus
        )

        self.state = self.integrator.scheme(
            self.state,
            self.model.dfun,
            self.coupling,
            self.local_coupling,
            stimulus,
        )
        return self.state