"""
All the bandpass filters to produce the reward signal
- Calculates  the LFP form the epileptor.py
- Applies the band pass filters
- State estimator
- Reward (epileptiform activity (calculated with filters)-the cost of the stimulation)

"""
import math

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi



class ButterFilter:
    def __init__(self, btype, cutoff, fs=1000.0, order=1):
        self.sos = butter(order, cutoff, btype=btype, fs=fs, output="sos")
        self.zi = None

    def push(self, x):
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos) * x
        y, self.zi = sosfilt(self.sos, [x], zi=self.zi)
        return y[0]

    def reset(self):
        self.zi = None

    def reset(self):
        self.y = 0.0


def lfp(state):
    """
    The measurement of the state (local state potential)
    -x1+x2
    """
    
    s = np.ravel(state)
    return -s[0] + s[3]


class FeatureExtractor:
    def __init__(self, fs=1000.0):
        self.lowPass = ButterFilter("lowpass", 0.5, fs)
        self.highPass = ButterFilter("highpass", 0.5, fs)
        self.bandPass = ButterFilter("bandpass", [2.0, 15.0], fs)
        self.smoother = ButterFilter("lowpass", 0.02, fs)

        #for agent decisions
        self.last_low = 0.0
        self.last_high = 0.0
        self.last_power = 0.0

        self.threshold = 0.0
        self.ictal = False

    def push(self, state):
        """
        called per every simulation step
        """
        x = lfp(state)

        #state estimate signal parts (low and high freq)
        self.last_low = self.lowPass.push(x)
        self.ictal = self.last_low < self.threshold
        self.last_high = self.highPass.push(x)

        #reward signal part
        band = self.bandPass.push(x)
        self.last_power = self.smoother.push(band * band)

    def read_out(self):
        """
        Returns (state, activity)
        """
        return self.get_state(), self.last_power

    def get_state(self):
        return int(self.ictal)

    def reset(self):
        """
        Resets the filter memory (in case we need it at some point)
        """
        self.lowPass.reset()
        self.highPass.reset()
        self.bandPass.reset()
        self.smoother.reset()

        self.last_low = 0.0
        self.last_high = 0.0
        self.last_power = 0.0
        self.ictal = False



