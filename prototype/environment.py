"""
Takes an action from agent.py and executes 15 seconds of stimulation (15 000 steps)
and gives the requested pulse to the state at (0, 500, 1000, ...) and reports how good it was
- based on the chosen action by the agent we apply the stimulus
- advance the simulation
- Need to calculate the reward here: reward = -log(smoothed Power (from features) - cost(based on the
actions, formula from the paper))
"""

import numpy as np
import math
from tvb.simulator.integrators import EulerStochastic
from tvb.simulator.noise import Additive
from tvb.simulator.models.epileptor import Epileptor
from prototype.epileptor import NagarajEpileptor
from prototype.features import lfp, FeatureExtractor
from prototype.plot import plot_state_space, plot_seizure_trajectory, plot_seizure_signals
from prototype.config import CONFIGS, SimulationConfig
from prototype.simulator import Simulator

config = CONFIGS["fast"]


def print_state(state, step):
    print(
        f"t = {(step) * simulator.config.step_size:.3f} ms | "
        f"x1={state[0, 0, 0]:.6f}, "
        f"y1={state[1, 0, 0]:.6f}, "
        f"z={state[2, 0, 0]:.6f}, "
        f"x2={state[3, 0, 0]:.6f}, "
        f"y2={state[4, 0, 0]:.6f}, "
        f"u={state[5, 0, 0]:.6f}"
    )

    print("-" * 80)
    if not np.all(np.isfinite(state)):
        raise RuntimeError(f"Non-finite state at step {step}")

def run_simulation(config: SimulationConfig):

    simulator = Simulator(config)
    print("Initial state:", simulator.state)
    print("Total steps:", simulator.total_steps)
    #Initialize return arrays for LFP and features
    fe = FeatureExtractor(fs=config.fs)
    low, high, last_power = [], [], [] 
    x1, x2, z, lfp_values = [], [], [], []

    s = 0
    steps_per_sampple = int(config.sampling_interval / config.step_size)
    for step in range(simulator.total_steps):
        state = simulator.next_state()

        # Generate LFP every 1 ms
        if (step+1) % steps_per_sampple == 0:
            print(f"{s}/15000")
            s+=1
            # Get LFP signal from the current state
            signal = lfp(state)
            lfp_values.append(signal)
            # Append state for data
            x1.append(state[0, 0, 0])
            x2.append(state[3, 0, 0])
            z.append(state[2, 0, 0])

            # Update filters and state estimate
            fe.push(state)
            # Append filtered signal for data
            low.append(fe.last_low)
            high.append(fe.last_high)
            last_power.append(fe.last_power)
            

        # Apply scheduled stimulation pulse to z

        # return filtered data
    print("Expected samples:", config.simulation_length)
    print("LFP samples:", len(lfp_values))
    print("Feature samples:", len(low), len(high), len(last_power))
    print("z samples:", len(z))
    return {
        "time": np.arange(1, len(lfp_values) + 1) * config.sampling_interval,
        "low": np.array(low),
        "high": np.array(high),
        "last_power": np.array(last_power),
        "x1": np.array(x1),
        "x2": np.array(x2),
        "z": np.array(z),
        "lfp": np.array(lfp_values),
    }

data = run_simulation(config)

# plot_state_space(data)
plot_seizure_signals(data)

# reward function 
# smoothed_power - LPF(bp), amplitude - Sa, ipi_seconds - simulation inter-pulse interval
# Note: if we use a negative weight we can also do log(smoothed_power - cost) 
# or we can do log(smoothed_power + cost) 

def calculate_reward(smoothed_power,amplitude, ipi_seconds, cw ):
    if amplitude == 0:
        cost = 0.0
    else:
        if ipi_seconds <= 0:
            raise ValueError("Stimulation interval must be positive")
        cost = cw * (amplitude**2 * (1/ipi_seconds))

    return -math.log(max(smoothed_power, 1e-12)) - cost 
