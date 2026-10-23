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

time_step = 0.01       #1 ms, Nagaraj (2017)
simulation_length = 15000 # ms, 15 seconds
fs = 1.0 / time_step

def print_state(state, step):
    print(
        f"t = {(step) * time_step:.3f} ms | "
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

def run_simulation(simulation_length = 15000, time_step = 0.001, fs=1000.0):

    total_steps = int(simulation_length / time_step)
    print(f"Running simulation for {simulation_length} ms with {total_steps} steps of {time_step} ms each.")

    model = Epileptor()
    model.configure()
    print("Original r:", model.r)

    # Change r = 1/\tau0, tau0 = 800 ms, Nagaraj(2017)
    # or r = 0.00035 Jirsa(2014)
    model.r = np.array([0.00125]) 

    # TVB integrator dt is in milliseconds
    # define noise according to Jirsa(2014)
    noise_nsig = np.array([
        0.0,  # x1
        0.0,  # y1
        0.0,    # z
        1e-3,   # x2
        1e-3,   # y2
        0.0,    # g
    ]).reshape(6, 1, 1)
    integrator = EulerStochastic(
        dt=time_step,
        noise=Additive(nsig=noise_nsig, noise_seed=42),
    )
    integrator.configure()
    integrator.noise.dt = integrator.dt

    # State shape: (variables, nodes, modes)
    state = np.zeros((6, 1, 1))
    state[:, 0, 0] = [0.0, -5.0, 3.0, 0.0, 0.0, 0.0]  # t(0) state, Jirsa(2014)

    coupling = np.zeros((1, 1, 1))
    local_coupling = 0.0
    stimulus = np.zeros_like(state)

    #Initialize return arrays for LFP and features
    fe = FeatureExtractor(fs=fs)
    low, high, last_power, x1, x2, z = [], [], [], [], [], []
    lfp_values = []

    s = 0
    for step in range(total_steps):
        state = integrator.scheme(
            state, model.dfun, coupling, local_coupling, stimulus
        )


        # Generate LFP every 1 ms
        if step % int(1.0 / time_step) == 0:
            print(f"{s}/15000")
            s+=1
            signal = lfp(state)
            lfp_values.append(signal)
            # Update filters and state estimate
            fe.push(state)
            low.append(fe.last_low)
            high.append(fe.last_high)
            last_power.append(fe.last_power)
            x1.append(state[0, 0, 0])
            x2.append(state[3, 0, 0])
            z.append(state[2, 0, 0])

        # Apply scheduled stimulation pulse to z

        # return filtered data
    print("Expected samples:", simulation_length)
    print("LFP samples:", len(lfp_values))
    print("Feature samples:", len(low), len(high), len(last_power))
    print("z samples:", len(z))
    return {
        "time": np.arange(simulation_length) * time_step,
        "low": np.array(low),
        "high": np.array(high),
        "last_power": np.array(last_power),
        "x1": np.array(x1),
        "x2": np.array(x2),
        "z": np.array(z),
        "lfp": np.array(lfp_values),
    }

data = run_simulation(simulation_length=simulation_length, time_step=time_step)

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
