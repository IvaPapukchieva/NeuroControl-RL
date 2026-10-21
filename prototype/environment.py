"""
Takes an action from agent.py and executes 15 seconds of stimulation (15 000 steps)
and gives the requested pulse to the state at (0, 500, 1000, ...) and reports how good it was
- based on the chosen action by the agent we apply the stimulus
- advance the simulation
- Need to calculate the reward here: reward = -log(smoothed Power (from features) - cost(based on the
actions, formula from the paper))
"""

import numpy as np
from tvb.simulator.integrators import EulerStochastic
from tvb.simulator.noise import Additive
from prototype.epileptor import NagarajEpileptor
from tvb.simulator.models.epileptor import Epileptor

time_step = 0.001       #1 ms, Nagaraj (2017)
simulation_length = 15000 # ms, 15 seconds

model = Epileptor()
model.configure()

# TVB integrator dt is in milliseconds
integrator = EulerStochastic(
    dt=time_step,
    noise=Additive(nsig=np.array([0.0]))
)
integrator.configure()
integrator.noise.dt = integrator.dt

print("Integrator dt:", integrator.dt)
print("Noise dt:", integrator.noise.dt)


# State shape: (variables, nodes, modes)
state = np.zeros((6, 1, 1))
state[:, 0, 0] = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]  # t(0) state

coupling = np.zeros((1, 1, 1))
local_coupling = 0.0
stimulus = np.zeros_like(state)

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

for step in range(simulation_length):
    state = integrator.scheme(
        state, model.dfun, coupling, local_coupling, stimulus
    )

    print_state(state, step)

    # Generate LFP and update RL variables here
    #lfp = generate_lfp(state)

    # Update filters and state estimate
    # Apply scheduled stimulation pulse to z