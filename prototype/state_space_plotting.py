"""
Plotting the state space (low pass against high pass filter) to get the treshold for  the state estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from prototype.epileptor import NagarajEpileptor
from prototype.features import FeatureExtractor

model = NagarajEpileptor()
model.configure()

dt = 0.0001
seconds = 300

#state shape
state = np.array([-1.5, -10.0, 3.0, -0.9, 0.0, 0.0]).reshape(6, 1, 1)
coupling = np.zeros((1, 1, 1))

fe = FeatureExtractor()
low, high, x1, x2 = [], [], [], []

for step in range(int(seconds / dt)):
    state = state + dt * model.dfun(state, coupling)  #Euler step

    if not np.all(np.isfinite(state)):
        raise RuntimeError(f"Model blew up at t = {step * dt:.3f} s")

    if step % 10 == 0:
        fe.push(state)
        low.append(fe.last_low)
        high.append(fe.last_high)
        x1.append(state[0, 0, 0])
        x2.append(state[3, 0, 0])

low = np.array(low)[2000:]  #waiting until the filters settle down
high = np.array(high)[2000:]

x1 = np.array(x1)[2000:]
x2 = np.array(x2)[2000:]

#plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.scatter(x1[::10], x2[::10], s=1)
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_title("True model state (paper Fig. 5B)")

ax2.scatter(low[::10], high[::10], s=1)
ax2.set_xlabel("low-pass LFP")
ax2.set_ylabel("high-pass LFP")
ax2.set_title("Estimated from LFP (paper Fig. 5C)")

plt.tight_layout()
plt.show()
