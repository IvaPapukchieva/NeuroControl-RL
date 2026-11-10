import numpy as np

class ChizhovObserver:
    """QIF observer, Chizhov et al. (2018), Eqs. 9–10.

    Time: ms
    Voltage: mV
    Capacitance: pF
    Conductance: nS
    Current: pA
    """

    def __init__(self, gL_nS: float):
        if gL_nS <= 0:
            raise ValueError("gL_nS must be positive")

        self.gL = float(gL_nS)

        self.CU = 200.0       # pF
        self.gU = 0.4         # nS/mV
        self.U1 = -60.0      # mV
        self.U2 = -40.0      # mV
        self.VT = 25.0       # mV
        self.Vreset = -50.0  # mV

        self.U = -70.0       # mV

    def step(self, previous_state, next_state, dt, tau_m):
        """Advance observer using the population's effective input."""
        V0 = float(previous_state[2, 0, 0])
        V1 = float(next_state[2, 0, 0])

        # Reconstruct u/gL, including the stochastic increment.
        u_normalized = V0 + tau_m * (V1 - V0) / dt

        # nS * mV = pA
        u_pA = self.gL * u_normalized

        # pA / pF = mV/ms
        dU = (
            self.gU * (self.U - self.U1) * (self.U - self.U2)
            + u_pA
        ) / self.CU

        self.U += dt * dU

        spiked = self.U > self.VT
        if spiked:
            self.U = self.Vreset

        return self.U, spiked
