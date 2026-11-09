"""
Five equations plus the ODE integration (Euler–Maruyama integrator with a
time step of 1 ms) as step() that advances the model
- Takes x1,x2 blabla z
- Outputs the updated state
#CHANGE HERE means change to the original paper so  that the figures are  reproducible
"""

import numpy as np

from tvb.basic.neotraits._attr import NArray, Final, List
from tvb.simulator.models.base import Model

# Using the simplified Model instead of ModelNumbaDfun for testing purposes
class NagarajEpileptor(Model):
    """
    Epileptor model based on Nagaraj, Lamperski & Netoff (2017).

    State variables, in order:
        x1, y1, z, x2, y2, u
    """

    # Model parameters
    I1 = NArray(label="I1", default=np.array([3.1]))
    I2 = NArray(label="I2", default=np.array([0.45]))

    tau0 = NArray(label="tau0", default=np.array([800.0]))
    tau1 = NArray(label="tau1", default=np.array([0.005]))
    tau2 = NArray(label="tau2", default=np.array([0.01]))
    gamma = NArray(label="gamma", default=np.array([0.01]))

    x0 = NArray(label="x0", default=np.array([2.0]))

    state_variables = ("x1", "y1", "z", "x2", "y2", "u")
    _nvar = 6

    cvar = np.array([], dtype=np.int32)

    variables_of_interest = List(
        of=str,
        default=("x1", "z", "x2"),
    )

    state_variable_range = {
        "x1": np.array([-2.0, 1.0]),
        "y1": np.array([-10.0, 10.0]),
        "z":  np.array([0.0, 15.0]),
        "x2": np.array([-2.0, 2.0]),
        "y2": np.array([-10.0, 10.0]),
        "u":  np.array([-2.0, 2.0]),
    }

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        x1, y1, z, x2, y2, u = state_variables

        # Piecewise nonlinear functions from the paper
        f1 = np.where(
            x1 < 0.0,
            x1**3 - 3.0 * x1**2,
            #CHANGE HERE: added parentheses
            (x2 - 0.6 * (z - 4.0)**2) * x1,
        )

        f2 = np.where(
            x2 < -0.25,
            0.0,
            #CHANGE HERE:  deleated the *x1
            6.0 * (x2 + 0.25), #* x1,
        )

        # Slow variable target function
        h = self.x0 + 10.0 / (
            1.0 + np.exp((-x1 - 0.5) / 0.1)
        )

        # Differential equations
        #CHANGES HERE: I1 deleated from dz and added to dx1
        #dx1 = (y1 - f1 - z) / self.tau1
        dx1 = (y1 - f1 - z + self.I1) / self.tau1
        dy1 = (1.0 - 5.0 * x1**2 - y1) / self.tau1
        #dz = (h - z - self.I1) / self.tau0
        dz = (h - z) / self.tau0

        dx2 = (
            -y2 + x2 - x2**3
            + self.I2
            + 1.8 * u
            - 0.3 * (z - 3.5)
        ) / self.tau1

        dy2 = (-y2 + f2) / self.tau2
        du = -self.gamma * (u - 0.1 * x1)

        return np.array([dx1, dy1, dz, dx2, dy2, du])


class ChizhovEpileptor2(Model):
    # Parameters (time in ms)
    Na_0 = NArray(
        default=np.array([10.0]),
        label="Resting intracellular sodium concentration",
        doc="Baseline intracellular sodium concentration (mM).",
    )

    K_0 = NArray(
        default=np.array([3.0]),
        label="Initial extracellular potassium concentration",
        doc="Reference potassium concentration for the Nernst potential (mM).",
    )
    V_th = NArray(
        default=np.array([25.0]),
        label="Firing-rate activation threshold",
        doc="Membrane depolarization threshold of the firing-rate function (mV).",
    )

    # Time constants (ms)
    tau_K = NArray(
        default=np.array([100_000.0]),
        label="Extracellular potassium relaxation time",
        doc="Time constant for potassium diffusion and buffering (ms).",
    )

    tau_Na = NArray(
        default=np.array([20_000.0]),
        label="Intracellular sodium relaxation time",
        doc="Time constant for sodium leakage and diffusion (ms).",
    )

    tau_m = NArray(
        default=np.array([10.0]),
        label="Membrane time constant",
        doc="Population membrane polarization time constant (ms).",
    )

    tau_D = NArray(
        default=np.array([2_000.0]),
        label="Synaptic depression recovery time",
        doc="Time constant for synaptic resource recovery (ms).",
    )

    # Ionic concentrations (mM)
    K_bath = NArray(
        default=np.array([8.5]),
        label="Bath potassium concentration",
        doc="Extracellular potassium concentration in the bath (mM).",
    )


    # Ionic dynamics
    gamma = NArray(
        default=np.array([10.0]),
        label="Intracellular-to-extracellular volume ratio",
        doc="Volume ratio scaling potassium transport by the Na/K pump.",
    )

    rho = NArray(
        default=np.array([0.2]),
        label="Maximum Na/K pump flux",
        doc="Maximum sodium-potassium pump flux (mM/s).",
    )

    delta_K = NArray(
        default=np.array([0.02]),
        label="Potassium concentration increment per spike",
        doc="Extracellular potassium increment associated with neuronal firing (mM).",
    )

    delta_Na = NArray(
        default=np.array([0.03]),
        label="Sodium concentration increment per spike",
        doc="Intracellular sodium increment associated with neuronal firing (mM).",
    )

    delta_D = NArray(
        default=np.array([0.01]),
        label="Synaptic resource depletion per spike",
        doc="Fractional depletion of available synaptic resources per spike.",
    )

    # Population firing-rate function
    nu_max = NArray(
        default=np.array([100.0]),
        label="Maximum population firing rate",
        doc="Upper bound of the population firing-rate function (Hz).",
    )


    k_nu = NArray(
        default=np.array([20.0]),
        label="Firing-rate sigmoid slope parameter",
        doc="Controls the steepness of the voltage-to-firing-rate relationship (mV).",
    )

    # Membrane and synaptic coupling
    gK_ratio = NArray(
        default=np.array([0.5]),
        label="Relative potassium leak conductance",
        doc="Ratio of potassium leak conductance to total leak conductance (gK_leak/gL).",
    )

    Gsyn_ratio = NArray(
        default=np.array([5.0]),
        label="Normalized synaptic coupling strength",
        doc="Postsynaptic charge normalized by leak conductance (mV*s).",
    )

    # TVB model metadata
    state_variables = ("K_o", "Na_i", "V", "x_D")

    _nvar = 4
    nvar = Final(default=4)

    cvar = np.array([2], dtype=np.int32)

    variables_of_interest = List(
        of=str,
        choices=state_variables,
        default=("K_o", "Na_i", "V", "x_D"),
    )

    state_variable_range = Final(
        default={
            "K_o": np.array([3.0, 12.0]),
            "Na_i": np.array([10.0, 25.0]),
            "V": np.array([-10.0, 100.0]),
            "x_D": np.array([0.0, 1.0]), # synaptic resource
        }
    )

    def dfun(self, state, coupling, local_coupling=0.0):
        K, Na, V, D = state

        # Firing rate (Hz)
        nu = self.nu_max * np.maximum(
            2.0 / (1.0 + np.exp(-2.0 * (V - self.V_th) / self.k_nu))
            - 1.0,
            0.0,
        )

        # Nernst potential difference (mV) (simplified math)
        dVK = 26.6 * np.log(K / self.K_0)

        # Na/K pump (mM/s)
        pump = self.rho / (
            (1.0 + np.exp(3.5 - K))
            * (1.0 + np.exp((25.0 - Na) / 3.0))
        )

        # Population input (mV) == u(t)/gL
        normalized_input = (
            self.gK_ratio * dVK
            + self.Gsyn_ratio * nu * (D - 0.5)
        )

        # Convert ionic rates from mM/s to mM/ms
        dK = (
            (self.K_bath - K) / self.tau_K
            - 2.0 * self.gamma * pump / 1000.0
            + self.delta_K * nu / 1000.0
        )

        dNa = (
            (self.Na_0 - Na) / self.tau_Na
            - 3.0 * pump / 1000.0
            + self.delta_Na * nu / 1000.0
        )

        dV = (-V + normalized_input) / self.tau_m

        dD = (
            (1.0 - D) / self.tau_D
            - self.delta_D * D * nu / 1000.0
        )

        return np.array([dK, dNa, dV, dD])
