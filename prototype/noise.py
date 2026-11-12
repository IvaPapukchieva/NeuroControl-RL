import numpy as np

from tvb.simulator.noise import Noise

class ChizhovNoise(Noise):
    """
    Sampled Gaussian forcing for Chizhov Epileptor-2.

    Time: ms
    Voltage: mV

    Generates a voltage increment for EulerStochastic,
    rather than a Wiener increment.
    """

    def __init__(
        self,
        noise_strength=30.0,
        tau_m=10.0,
        sample_interval=1.0,
        duration_ms=1_000_000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.noise_strength = float(noise_strength)
        self.tau_m = float(tau_m)
        self.sample_interval = float(sample_interval)

        if self.tau_m <= 0 or self.sample_interval <= 0:
            raise ValueError("Time constants must be positive")

        n = int(np.ceil(duration_ms / self.sample_interval)) + 3

        # Gaussian samples of eta(t)/gL, in mV.
        self.samples = self.random_stream.normal(
            0.0, self.noise_strength, size=n
        )

        # Mathematica prepends a zero sample at t=0.
        self.samples[0] = 0.0

        # Approximate Hermite derivatives (mV/ms).
        self.slopes = np.gradient(
            self.samples, self.sample_interval
        )

        self._step = 0

    def gfun(self, state_variables):
        # The generated values are already voltage increments.
        # No sqrt(2*nsig) factor.
        gain = np.zeros(
            (state_variables.shape[0],) +
            (1,) * (state_variables.ndim - 1)
        )
        gain[2] = 1.0
        return gain

    def generate(self, shape, lo=-1.0, hi=1.0):
        if self.dt is None:
            raise RuntimeError("Noise dt has not been configured")

        t = self._step * self.dt
        self._step += 1

        position = t / self.sample_interval
        i = int(position)
        if i + 1 >= len(self.samples):
            raise RuntimeError("Noise duration exceeded")

        s = position - i

        y0 = self.samples[i]
        y1 = self.samples[i + 1]

        m0 = self.slopes[i] * self.sample_interval
        m1 = self.slopes[i + 1] * self.sample_interval

        # Cubic Hermite basis functions
        h00 = 2*s**3 - 3*s**2 + 1
        h10 = s**3 - 2*s**2 + s
        h01 = -2*s**3 + 3*s**2
        h11 = s**3 - s**2

        eta = h00*y0 + h10*m0 + h01*y1 + h11*m1

        # Convert forcing (mV) to voltage increment (mV).
        dV_noise = (self.dt / self.tau_m) * eta

        result = np.zeros(shape, dtype=float)
        result[2, ...] = dV_noise

        return result

    def reset_random_stream(self):
        raise NotImplementedError(
            "Create a new ChizhovNoise instance to restart the noise sequence"
        )