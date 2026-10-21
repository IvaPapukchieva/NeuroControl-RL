"""
Five equations plus the ODE integration (Euler–Maruyama integrator with a
time step of 1 ms) as step() that advances the model
- Takes x1,x2 blabla z
- Outputs the updated state
"""

import numpy as np

from tvb.basic.neotraits._attr import NArray, List
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
            x2 - 0.6 * (z - 4.0)**2 * x1,
        )

        f2 = np.where(
            x2 < -0.25,
            0.0,
            6.0 * (x2 + 0.25) * x1,
        )

        # Slow variable target function
        h = self.x0 + 10.0 / (
            1.0 + np.exp((-x1 - 0.5) / 0.1)
        )

        # Differential equations
        dx1 = (y1 - f1 - z) / self.tau1
        dy1 = (1.0 - 5.0 * x1**2 - y1) / self.tau1
        dz = (h - z - self.I1) / self.tau0

        dx2 = (
            -y2 + x2 - x2**3
            + self.I2
            + 1.8 * u
            - 0.3 * (z - 3.5)
        ) / self.tau1

        dy2 = (-y2 + f2) / self.tau2
        du = -self.gamma * (u - 0.1 * x1)

        return np.array([dx1, dy1, dz, dx2, dy2, du])