import numpy as np
from prototype.epileptor import ChizhovEpileptor2

model = ChizhovEpileptor2()
model.configure()

# Shape: (state variables, nodes, modes)
state = np.array([
    [[3.0]],   # K_o  (mM)
    [[10.0]],  # Na_i (mM)
    [[0.0]],   # V    (mV) - assumed zero initial membrane potential
    [[1.0]],   # x_D  (dimensionless) - assumed fully recovered synaptic resources
])

# No network coupling
coupling = np.zeros((1, 1, 1))

derivatives = model.dfun(state, coupling)

print("Variables:", model.state_variables)
print("Derivatives:")

for name, value in zip(model.state_variables, derivatives):
    print(f"  d{name}/dt = {value.item():.8f}")

# Structural validation
assert derivatives.shape == state.shape
assert np.isfinite(derivatives).all()

# Physiological sanity checks
assert derivatives[0, 0, 0] > 0  # K_o approaches K_bath
assert derivatives[2, 0, 0] == 0  # V = 0, no initial drive
assert derivatives[3, 0, 0] == 0  # Full synaptic resources

print("Epileptor-2 derivative test passed.")