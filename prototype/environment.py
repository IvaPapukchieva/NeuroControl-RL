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
from prototype.features import FeatureExtractor
from prototype.plot import (plot_state_space, plot_seizure_trajectory, plot_seizure_signals, plot_observed_signal, 
                            plot_population_diagnostics, diagnose_chizhov, plot_fig3, plot_synaptic_resource, plot_kbath,
                            plot_chizhov_phase_portrait)
from prototype.config import CONFIGS, SimulationConfig
from prototype.simulator import Simulator

config = CONFIGS["chizhov_2018_fast"]


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

def run_simulation(config: SimulationConfig, potassium_ramp: bool = False):

    simulator = Simulator(config)
    print("Initial state:", simulator.state)
    print("Total steps:", simulator.total_steps)

    for name in (
        "tau_K", "tau_Na", "tau_m", "tau_D",
        "K_bath", "rho", "gamma",
        "delta_K", "delta_Na", "delta_D",
        "nu_max", "V_th", "k_nu",
        "gK_ratio", "Gsyn_ratio",
    ):
        print(f"{name:15s} = {getattr(simulator.model, name)}")

    print("dt =", simulator.integrator.dt)
    noise = simulator.integrator.noise

    print("noise type =", type(noise).__name__)
    print("noise strength =", noise.noise_strength)
    print("noise seed =", noise.noise_seed)
    print("noise sample interval (ms) =", noise.sample_interval)
    observer = simulator.observer

    observer_voltage = []
    observer_spikes = []
    observer_spike_times = []

    #Initialize return arrays for LFP and features
    fe = FeatureExtractor(fs=config.fs)


    s = 0
    steps_per_sample = round(config.sampling_interval / config.step_size)

    if not np.isclose(
        steps_per_sample * config.step_size,
        config.sampling_interval,
    ):
        raise ValueError(
            "sampling_interval must be a multiple of step_size"
        )

    times = []
    states = []
    lfp_values = []
    low, high, last_power = [], [], []

    # Define stimulus to z:
    stimulus_strength = 0.0
    stimulus = np.zeros_like(simulator.state)

    for step in range(simulator.total_steps):

        # Reset stimulation every integration step
        stimulus.fill(0.0)
        # Apply scheduled stimulation pulse to z
        # Example: Every second we apply a stimulus pulse to z for 10 ms (10 steps)
        # if sample_timer < 10:
        #     stimulus[2, 0, 0] = stimulus_strength # apply stimulus to z
        
        previous_state = simulator.state.copy()
        if potassium_ramp:
            t_ms = step * config.step_size
            fraction = np.clip(t_ms / 200_000.0, 0.0, 1.0)

            simulator.model.K_bath = np.array([
                3.0 + 19.0 * fraction
            ])
        state = simulator.next_state(
            new_stimulus=stimulus
        )
        t = (step + 1) * config.step_size

        # Update observer at every integration step
        if observer is not None:
            U, spiked = observer.step(
                previous_state=previous_state,
                next_state=state,
                dt=config.step_size,
                tau_m=float(simulator.model.tau_m[0]),
            )

            if spiked:
                observer_spike_times.append(t)


        # Generate LFP every 1 ms
        if (step + 1) % steps_per_sample != 0:
            continue

        if observer is not None:
            observer_voltage.append(U)
            observer_spikes.append(spiked)

        #print(f"{s}/{config.simulation_length}")
        s+=1

        # Get LFP signal from the current state
        signal = U if observer is not None else lfp(state)

        times.append(t)
        states.append(state.copy())
        lfp_values.append(signal)

        # Update filters and state estimate
        fe.push(signal)

        # Append filtered signal for data
        low.append(fe.last_low)
        high.append(fe.last_high)
        last_power.append(fe.last_power)
            


    print("Expected samples:", config.simulation_length)
    print("LFP samples:", len(lfp_values))
    print("Feature samples:", len(low), len(high), len(last_power))
    # return filtered data
    return {
        "time": np.asarray(times),
        "states": np.stack(states),
        "state_variables": simulator.model.state_variables,
        "model": simulator.model,
        "lfp": np.asarray(lfp_values),
        "low": np.asarray(low),
        "high": np.asarray(high),
        "power": np.asarray(last_power),
        "observer_voltage": (
            np.asarray(observer_voltage) if observer is not None else None
        ),
        "observer_spikes": (
            np.asarray(observer_spikes) if observer is not None else None
        ),
        "observer_spike_times": (
            np.asarray(observer_spike_times) if observer is not None else None
        ),
    }

data = run_simulation(config)

plot_chizhov_phase_portrait(data, data["model"])
#diagnose_chizhov(data)
plot_fig3(data)
plot_population_diagnostics(data)
plot_synaptic_resource(data)
#plot_seizure_signals(data)

#plot_state_space(data)
plot_seizure_trajectory(data)
plot_kbath(data)
plot_observed_signal(data, show_spikes=False)

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
