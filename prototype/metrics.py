"""
Evaluation protocol and metrics for comparing the agent with the baselines.

Seizures are counted from the ground truth (x1), not from the filtered LFP the
agent sees, so the metric is independent of the agent's own features.
Every controller (baselines and agent) is scored by these same functions.

What the environment has to record per run:
- x1 at FS (1 kHz), for the seizure count
- the number of stimulation pulses it actually delivered
- the reward and the band power (FeatureExtractor.last_power) at each decision

To score the agent's learning phase and its converged phase separately, pass
the corresponding slices of the recorded arrays to run_metrics.
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

# ---------------------------------------------------------------- protocol
# (to be moved to config.py once the group agrees on it)
SEEDS = list(range(10))             # the same seeds for every controller
RUN_LENGTH_S = 15000                # Nagaraj et al. (2017)
DECISION_WINDOW_S = 15              # Nagaraj et al. (2017): w = 15 s
FS = 1000.0                         # recording rate of x1 and the LFP (Hz)

# tau0 over time: (start time in s, tau0). The switch is the condition where
# adapting matters: a fixed frequency that works before it fails after it.
CONDITIONS = {
    "slow (tau0=800)": [(0, 800.0)],
    "fast (tau0=400)": [(0, 400.0)],
    "switch 800->400": [(0, 800.0), (RUN_LENGTH_S / 2, 400.0)],
}

# Derived from the paper's minima (1.04 Hz at tau0=800, 2.08 Hz at tau0=400) with
# delta_z_null = 2.91 - 2.0. Re-check with the open-loop sweep now that the model
# equations were corrected: 1.04 Hz should just suppress seizures at tau0 = 800.
PULSE_AMPLITUDE = 1.09e-3

MIN_SEIZURE_S = 1.0                 # shorter threshold crossings are not seizures
SETTLE_S = 10.0                     # start of each run ignored (initial conditions, filters)


def tau0_at(condition, t_s):
    """tau0 that applies at time t_s for a condition from CONDITIONS."""
    current = condition[0][1]
    for start, tau0 in condition:
        if t_s >= start:
            current = tau0
    return current


# ---------------------------------------------------------------- seizures
def smooth_x1(x1, fs=FS, cutoff_hz=0.5):
    """
    Low-pass x1 so only the slow DC shift of a seizure remains.
    Same first-order Butterworth as features.py, applied to the whole recorded trace at once.
    """
    x1 = np.asarray(x1, dtype=float)
    sos = butter(1, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    smoothed, _ = sosfilt(sos, x1, zi=sosfilt_zi(sos) * x1[0])
    return smoothed


def drop_settling(trace, fs=FS, settle_s=SETTLE_S):
    """Remove the first settle_s seconds of a per-sample trace."""
    return np.asarray(trace)[int(settle_s * fs):]


def seizure_threshold(smoothed_x1_no_stim, fs=FS):
    """
    Set once from a no-stimulation run, then fixed for every controller.
    Halfway between the non-seizing level (1st percentile) and the seizing level
    (99th percentile) of smoothed x1, ignoring the settling period. x1 is higher
    during seizures (the LFP, -x1 + x2, drops), so seizing means above this value.
    Check it against a plot of the trace before relying on it.
    """
    trace = drop_settling(smoothed_x1_no_stim, fs)
    low, high = np.percentile(trace, [1, 99])
    return float((low + high) / 2)


def seizure_mask(smoothed_x1, threshold, fs=FS, min_duration_s=MIN_SEIZURE_S):
    """True at every sample that belongs to a seizure lasting at least min_duration_s."""
    above = np.asarray(smoothed_x1) > threshold
    mask = np.zeros_like(above)
    min_samples = int(min_duration_s * fs)

    start = None
    for i, value in enumerate(np.append(above, False)):   # trailing False closes the last run
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_samples:
                mask[start:i] = True
            start = None
    return mask


def count_seizures(mask):
    """Number of separate seizures (rising edges) in a seizure mask."""
    mask = np.asarray(mask, dtype=int)
    return int(mask[0] + np.sum(np.diff(mask) == 1)) if mask.size else 0


def window_labels(mask, fs=FS, window_s=DECISION_WINDOW_S):
    """
    One label per decision window: True if more than half of the window was seizing.
    Use this to turn a per-sample seizure_mask into the per-window labels that
    baseline.detector_threshold expects (band power is read once per window).
    """
    samples_per_window = int(window_s * fs)
    n_windows = len(mask) // samples_per_window
    windows = np.asarray(mask[:n_windows * samples_per_window]).reshape(n_windows, samples_per_window)
    return windows.mean(axis=1) > 0.5


# ---------------------------------------------------------------- energy
def stimulation_energy(n_pulses, amplitude=PULSE_AMPLITUDE):
    """
    Total energy = amplitude^2 x number of pulses delivered.
    Nagaraj et al. (2017) define energy as amplitude^2 x stimulation duration;
    with fixed 1 ms pulses that is proportional to the pulse count.
    """
    return float(amplitude**2 * n_pulses)


# ---------------------------------------------------------------- one run and summary
def run_metrics(smoothed_x1, threshold, n_pulses, rewards, fs=FS):
    """
    Metrics for one run of one controller.
    smoothed_x1 should already have the settling period removed (drop_settling).
    """
    mask = seizure_mask(smoothed_x1, threshold, fs)
    return {
        "time_seizing_pct": 100.0 * float(mask.mean()),
        "n_seizures": count_seizures(mask),
        "energy": stimulation_energy(n_pulses),
        "mean_reward": float(np.mean(rewards)),
    }


def summarize(runs):
    """Mean and SD over seeds: {metric: (mean, sd)} from a list of run_metrics dicts."""
    summary = {}
    for key in runs[0]:
        values = [r[key] for r in runs]
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary[key] = (float(np.mean(values)), sd)
    return summary
