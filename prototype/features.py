"""
All the bandpass filters to produce the reward signal
- Calculates  the LFP form the epileptor.py
- Applies the band pass filters
- State estimator
- Reward (epileptiform activity (calculated with filters)-the cost of the stimulation)

"""
import math

import numpy as np


class Lowpass:
    """
    A low pass filter, that remembers the previous states
    """
    def __init__(self, fs=1000, cutoff=0.5):
        self.a = math.exp(-2*math.pi*cutoff/fs)
        self.y = 0.0

    def push(self,x):
        """
        advances the filter results throughout the simulation
        """
        self.y = self.a * self.y + (1 - self.a) * x
        return self.y

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
        self.fs = fs
        self.lowPass = Lowpass(fs, 0.5)
        self.highPass_helper = Lowpass(fs, 0.5)

        #for the band pass
        self.bp_low = Lowpass(fs, 2.0)
        self.bp_high = Lowpass(fs, 15.0)

        self.smoother = Lowpass(fs, 0.02)

        #for agent decisions
        self.last_low = 0.0
        self.last_high = 0.0
        self.last_power = 0.0

    def push(self, state):
        """
        called per every simulation step
        """
        x = lfp(state)

        #state parts (low and high freq)
        self.last_low = self.lowPass.push(x)
        self.last_high = x - self.highPass_helper.push(x)

        #reward signal part
        band = self.bp_high.push(x - self.bp_low.push(x))
        self.last_power = self.smoother.push(band * band)

    def read_out(self):
        """
        Returns (state, activity)
        """
        return self.get_state(), self.last_power

    def get_state(self):
        #todo
        raise NotImplementedError

    def reset(self):
        """
        Resets the filter memory (in case we need it at some point)
        """
        self.lowPass.reset()
        self.highPass_helper.reset()
        self.bp_low.reset()
        self.bp_high.reset()
        self.smoother.reset()

        self.last_low = 0.0
        self.last_high = 0.0
        self.last_power = 0.0



