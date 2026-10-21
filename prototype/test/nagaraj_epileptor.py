import numpy as np
from prototype.epileptor import NagarajEpileptor

model = NagarajEpileptor()
model.configure()

# Shape: (state variables, nodes, modes)
state = np.array([
    [[-1.0]],  # x1
    [[ 0.0]],  # y1
    [[ 3.0]],  # z
    [[ 0.0]],  # x2
    [[ 0.0]],  # y2
    [[ 0.0]],  # u
])

# No network coupling for this single-node test
coupling = np.zeros((0, 1, 1))

derivatives = model.dfun(state, coupling)

print("Variables:", model.state_variables)
print("Derivatives:")
for name, value in zip(model.state_variables, derivatives):
    print(f"  d{name}/dt = {value.item():.6f}")

assert derivatives.shape == state.shape
assert np.isfinite(derivatives).all()

print("Model derivative test passed.")