states = data["states"]

V = states[:, 2, 0, 0]
D = states[:, 3, 0, 0]
t = data["time"] / 1000.0

nu = 100.0 * np.maximum(
    np.tanh((V - 25.0) / 20.0),
    0.0,
)

mask = t > 50

print("Mean firing rate after 50 s:", nu[mask].mean())
print("Mean synaptic resource:", D[mask].mean())
print("Fraction of time firing:", np.mean(nu[mask] > 0))