"""
Non-learning controllers to compare the agent against.

Every controller has the same interface as the agent:
    act(state, activity) -> stimulation frequency in Hz for the next decision window
    (0 means no stimulation)
where `state` and `activity` are what FeatureExtractor.read_out() returns.
Controllers never learn, so the environment loop stays identical: only the
controller object is swapped.

Controllers (all use the same fixed pulse amplitude as the agent):
- NoStimulation        untreated seizure pattern, the lower bound
- OpenLoop             constant frequency, ignores the brain state
- DetectorTriggered    fixed frequency only while band power is above a threshold
- ShuffledAgentActions the agent's own frequencies in random order (same energy, random timing)
"""

import numpy as np

# Frequencies the agent can choose from (Hz), Nagaraj et al. (2017) Fig. 8
ACTION_FREQUENCIES_HZ = [0, 1, 2, 3, 4, 5]

# Lowest whole-number frequency that suppresses seizures for both tau0 = 800 (min 1.04 Hz)
# and tau0 = 400 (min 2.08 Hz), see Nagaraj et al. (2017) Section 2.2
DETECTOR_FREQUENCY_HZ = 3


class NoStimulation:
    """Never stimulates."""

    name = "no stimulation"

    def act(self, state, activity=None):
        return 0


class OpenLoop:
    """Stimulates at the same frequency in every window, whatever the brain is doing."""

    def __init__(self, frequency_hz):
        self.frequency_hz = frequency_hz
        self.name = f"open-loop {frequency_hz} Hz"

    def act(self, state, activity=None):
        return self.frequency_hz


class DetectorTriggered:
    """
    Stimulates only when seizure activity is detected, like responsive neurostimulation.
    Decides once per window, like the agent, so it has the same information and timing.
    """

    def __init__(self, power_threshold, frequency_hz=DETECTOR_FREQUENCY_HZ):
        self.power_threshold = power_threshold
        self.frequency_hz = frequency_hz
        self.name = f"detector-triggered {frequency_hz} Hz"

    def act(self, state, activity):
        if activity > self.power_threshold:
            return self.frequency_hz
        return 0


class ShuffledAgentActions:
    """
    Replays the frequencies the trained agent chose, but in a random order.
    Total energy is exactly the agent's; only the timing is random.
    If the agent does not beat this, its timing adds nothing.
    """

    def __init__(self, agent_frequencies_hz, seed):
        rng = np.random.default_rng(seed)
        self.schedule = rng.permutation(agent_frequencies_hz).tolist()
        self.window = 0
        self.name = "shuffled agent actions"

    def act(self, state, activity=None):
        frequency = self.schedule[self.window % len(self.schedule)]
        self.window += 1
        return frequency


def open_loop_sweep():
    """One open-loop controller per non-zero frequency the agent can choose."""
    return [OpenLoop(f) for f in ACTION_FREQUENCIES_HZ if f > 0]


def detector_threshold(band_power, seizure_mask):
    """
    Threshold for DetectorTriggered, set once from a no-stimulation run and then fixed.

    band_power:   smoothed 2-15 Hz power per window (what the detector sees)
    seizure_mask: True for windows that were seizing (ground truth, see metrics.py)

    Returns the midpoint in log space between the median power during and
    between seizures (power spans orders of magnitude, so the log midpoint is fairer).
    """
    band_power = np.asarray(band_power, dtype=float)
    seizure_mask = np.asarray(seizure_mask, dtype=bool)
    if seizure_mask.all() or not seizure_mask.any():
        raise ValueError("Need both seizing and non-seizing windows to set a threshold")

    log_power = np.log(np.maximum(band_power, 1e-12))
    ictal = np.median(log_power[seizure_mask])
    interictal = np.median(log_power[~seizure_mask])
    return float(np.exp((ictal + interictal) / 2))
