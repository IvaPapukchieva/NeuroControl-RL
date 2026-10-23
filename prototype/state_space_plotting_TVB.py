import numpy as np
import matplotlib.pyplot as plt
from tvb.simulator.models.epileptor import Epileptor
from prototype.features import FeatureExtractor

model = Epileptor()
model.configure()

dt = 0.001                          # TVB time units
steps = 15000                     # = 6000 time units, about 3 seizures
state = np.zeros((6, 1, 1))
state[2] = 3.0                     # same start as in environment.py
coupling = np.zeros((2, 1, 1))     # single node, no network

fe = FeatureExtractor(fs=1000.0 / dt)
low, high, x1, x2 = [], [], [], []

for step in range(steps):
    state = state + dt * model.dfun(state, coupling)   # Euler step
    fe.push(state)
    low.append(fe.last_low)
    high.append(fe.last_high)
    x1.append(state[0, 0, 0])
    x2.append(state[3, 0, 0])

# skip the start while the filters settle
low, high = np.array(low)[20000:], np.array(high)[20000:]
x1, x2 = np.array(x1)[20000:], np.array(x2)[20000:]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.scatter(x1[::10], x2[::10], s=1)
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_title("True model state")

ax2.scatter(low[::10], high[::10], s=1)
ax2.set_xlabel("low-pass LFP")
ax2.set_ylabel("high-pass LFP")
ax2.set_title("Estimated from LFP")

plt.tight_layout()
plt.show()