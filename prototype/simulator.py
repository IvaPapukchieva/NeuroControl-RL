import numpy as np

from tvb.simulator.models.epileptor import Epileptor
from tvb.simulator.integrators import EulerStochastic
from tvb.simulator.noise import Additive

from prototype.config import SimulationConfig
from prototype.epileptor import NagarajEpileptor, ChizhovEpileptor2
from prototype.features import lfp
from prototype.observer import ChizhovObserver

class Simulator:
    def __init__(self, config: SimulationConfig):
        self.config = config

        self.total_steps = int(
            config.simulation_length / config.step_size
        )

        self.model = config.model_class()

        self.observer = (
            ChizhovObserver(gL_nS=20.0)
            if isinstance(self.model, ChizhovEpileptor2)
            else None
        )

        # Apply model-specific parameters
        for name, value in config.model_params.items():
            if not hasattr(self.model, name):
                raise ValueError(f"Unknown model parameter: {name}")

            setattr(
                self.model,
                name,
                np.atleast_1d(value),
            )

        self.model.configure()

        # Infer model dimensions
        nvar = self.model.nvar

        self.integrator = EulerStochastic(
            dt=config.step_size,
            noise=Additive(
                nsig=config.noise_array(nvar),
                noise_seed=config.noise_seed,
            ),
        )
        self.integrator.configure()
        self.integrator.noise.dt = self.integrator.dt

        # State shape: (variables, nodes, modes)
        self.state = config.state_array(nvar).copy()

        # These 3 are not necessary for a single node simulation, but are included for completeness
        self.coupling = np.zeros(
            (len(self.model.cvar), 1, 1)
        )
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

    def observe_lfp(self, state):
        if isinstance(self.model, ChizhovEpileptor2):
            return float(state[2, 0, 0])  # Population voltage V

        if isinstance(self.model, Epileptor):
            return float(lfp(state))       # Existing Jirsa proxy

        raise NotImplementedError(
            f"No LFP observation defined for {type(self.model).__name__}"
        )