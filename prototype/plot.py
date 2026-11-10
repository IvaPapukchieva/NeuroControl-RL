import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_state_space(data):
    plt.figure(figsize=(8, 6))
    plt.plot(data["x1"], data["x2"], linewidth=1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("State-Space Trajectory")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_seizure_trajectory(
    data,
    variables=None,
    ax=None,
    cmap="viridis",
):
    """
    Plot a 3D state-space trajectory colored by simulation time.

    Expected data:
        states: (samples, variables, nodes, modes)
        state_variables: tuple of variable names
        time: simulation time in milliseconds
    """
    states = np.asarray(data["states"])
    names = tuple(data["state_variables"])
    time = np.asarray(data["time"]).ravel() / 1000.0

    if states.ndim != 4:
        raise ValueError(
            f"Expected 4D states, got {states.shape}"
        )

    if len(time) != states.shape[0]:
        raise ValueError("Time and state lengths do not match.")

    if len(time) < 2:
        raise ValueError("At least two samples are required.")

    # Select variables
    if variables is None:
        if {"x1", "x2", "z"}.issubset(names):
            variables = ("x1", "x2", "z")
        elif {"V", "K_o", "Na_i"}.issubset(names):
            variables = ("V", "K_o", "Na_i")
        else:
            variables = names[:3]

    if len(variables) != 3 or any(v not in names for v in variables):
        raise ValueError(
            f"Expected three valid variables; got {variables}"
        )

    x, y, z = (
        states[:, names.index(v), 0, 0].copy()
        for v in variables
    )

    labels = list(variables)

    # Preserve your original Jirsa convention
    if variables == ("x1", "x2", "z"):
        x = -x
        z = z + 5.0
        labels = [r"$-x_1$", r"$x_2$", r"$z+5$"]

    # Construct 3D line segments
    points = np.column_stack((x, y, z))
    segments = np.stack(
        (points[:-1], points[1:]),
        axis=1,
    )

    # Color each segment by its midpoint time
    segment_times = (time[:-1] + time[1:]) / 2

    norm = Normalize(vmin=time[0], vmax=time[-1])

    trajectory = Line3DCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=0.8,
    )
    trajectory.set_array(segment_times)

    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.add_collection3d(trajectory)

    # Explicit limits required for Line3DCollection
    for setter, values in (
        (ax.set_xlim, x),
        (ax.set_ylim, y),
        (ax.set_zlim, z),
    ):
        lo, hi = np.min(values), np.max(values)
        margin = max((hi - lo) * 0.05, 1e-6)
        setter(lo - margin, hi + margin)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title("3D Seizure Trajectory")

    cbar = fig.colorbar(
        trajectory,
        ax=ax,
        pad=0.12,
        shrink=0.7,
    )
    cbar.set_label("Simulation time (s)")

    fig.tight_layout()

    plt.show()
    return fig, ax


def plot_seizure_signals(data, figsize=(12, 8)):
    """
    Plot the observed signal, smoothed 2–15 Hz power,
    and a model-specific slow state variable.

    Expected data keys:
        time, lfp, power, states, state_variables
    """
    time = np.asarray(data["time"]).ravel()
    states = np.asarray(data["states"])
    names = tuple(data["state_variables"])

    # States shape: (samples, variables, nodes, modes)
    if states.ndim != 4:
        raise ValueError(
            f"Expected 4D states, got shape {states.shape}"
        )

    if states.shape[0] != len(time):
        raise ValueError("State and time lengths do not match.")

    # Select slow variable according to model
    if "z" in names:
        slow_var = "z"
        slow_label = "Slow variable z"
    elif "K_o" in names:
        slow_var = "K_o"
        slow_label = "Extracellular K⁺ (mM)"
    else:
        slow_var = names[0]
        slow_label = slow_var

    slow_index = names.index(slow_var)
    slow_values = states[:, slow_index, 0, 0]

    signals = [
        (np.asarray(data["lfp"]).ravel(), "Observed signal"),
        (np.asarray(data["power"]).ravel(), "Smoothed 2–15 Hz power"),
        (slow_values, slow_label),
    ]

    fig, axes = plt.subplots(
        len(signals), 1, figsize=figsize, sharex=True
    )

    # Convert milliseconds to seconds
    time_s = time / 1000.0

    for ax, (values, label) in zip(axes, signals):
        if len(values) != len(time):
            raise ValueError(
                f"'{label}' has {len(values)} samples, "
                f"but time has {len(time)}."
            )

        ax.plot(time_s, values, linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    plt.show()

def plot_observed_signal(data, ax=None, show_spikes=True):
    """Plot the Chizhov QIF observer membrane potential U(t)."""

    time = np.asarray(data["time"]) / 1000.0  # ms -> s
    voltage = np.asarray(data["observer_voltage"])

    if voltage.ndim != 1 or len(time) != len(voltage):
        raise ValueError("time and observer_voltage must be 1D arrays of equal length")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    ax.plot(time, voltage, color="steelblue", linewidth=0.7, label="U(t)")

    if show_spikes:
        spike_times = np.asarray(data["observer_spike_times"]) / 1000.0

        if spike_times.size:
            ax.vlines(
                spike_times,
                ymin=-50,
                ymax=25,
                color="crimson",
                alpha=0.5,
                linewidth=0.5,
                label="Spikes",
            )

    ax.set(
        xlabel="Time (s)",
        ylabel="Membrane potential (mV)",
        title="Chizhov Epileptor-2 — Observer neuron",
    )

    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    plt.show()
    return fig, ax

def plot_population_diagnostics(data):
    t = data["time"] / 1000.0
    states = data["states"]

    K = states[:, 0, 0, 0]
    Na = states[:, 1, 0, 0]
    V = states[:, 2, 0, 0]

    nu = 100.0 * np.maximum(
        np.tanh((V - 25.0) / 20.0),
        0.0,
    )

    # Sodium-potassium pump (Chizhov et al.)
    rho = 0.2
    I_pump = rho / (
        (1.0 + np.exp(3.5 - K))
        * (1.0 + np.exp((25.0 - Na) / 3.0))
    )

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True
    )

    axes[0].plot(t, nu, lw=0.5)
    axes[0].set_ylabel("Firing rate (Hz)")

    axes[1].plot(t, V, lw=0.5)
    axes[1].set_ylabel("Population V (mV)")

    # Ion concentrations (left axis)
    axes[2].plot(t, K, label=r"$K_o$", color="tab:blue")
    axes[2].plot(t, Na, label=r"$Na_i$", color="tab:orange")
    axes[2].set_ylabel("Concentration (mM)")
    axes[2].set_xlabel("Time (s)")

    # Pump current (right axis)
    ax_pump = axes[2].twinx()
    ax_pump.plot(
        t, I_pump,
        label=r"$I_{\mathrm{pump}}$",
        color="tab:green",
        lw=1.0,
    )
    ax_pump.set_ylabel(
        r"$I_{\mathrm{pump}}$ (mM/s)",
        color="tab:green",
    )
    ax_pump.tick_params(axis="y", labelcolor="tab:green")

    # Combined legend
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax_pump.get_legend_handles_labels()

    axes[2].legend(lines1 + lines2, labels1 + labels2)

    fig.tight_layout()
    fig.show()

    return fig, axes


def diagnose_chizhov(data):
    states = data["states"]
    V = states[:, 2, 0, 0]
    D = states[:, 3, 0, 0]
    t = data["time"] / 1000.0

    nu = 100.0 * np.maximum(
        np.tanh((V - 25.0) / 20.0), 0.0
    )

    mask = t > 50

    print("Mean firing rate (Hz):", nu[mask].mean())
    print("Mean synaptic resource:", D[mask].mean())
    print("Fraction of time firing:", np.mean(nu[mask] > 0))


def plot_fig3(data):

    t = data["time"] / 1000
    K = data["states"][:, 0, 0, 0]
    Na = data["states"][:, 1, 0, 0]
    V = data["states"][:, 2, 0, 0]
    D = data["states"][:, 3, 0, 0]

    nu = 100 * np.maximum(np.tanh((V - 25) / 20), 0)

    mask = (t >= 210) & (t <= 222)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 7))

    ax[0].plot(t[mask], K[mask])
    ax[0].set_ylabel(r"$[K]_o$ (mM)")

    ax[1].plot(t[mask], Na[mask])
    ax[1].set_ylabel(r"$[Na]_i$ (mM)")

    ax[2].plot(t[mask], nu[mask])
    ax[2].set_ylabel(r"$\nu$ (Hz)")
    ax[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()

def plot_synaptic_resource(data, start=None, end=None):
    t = data["time"] / 1000.0  # ms -> s
    D = data["states"][:, 3, 0, 0]

    mask = np.ones_like(t, dtype=bool)

    if start is not None:
        mask &= t >= start
    if end is not None:
        mask &= t <= end

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(t[mask], D[mask], color="tab:blue", lw=1)
    ax.axhline(
        0.5,
        color="tab:red",
        linestyle="--",
        label=r"$x_D = 0.5$",
    )

    ax.set(
        xlabel="Time (s)",
        ylabel=r"Synaptic resource $x_D$",
        title="Chizhov Epileptor-2 — Synaptic depression",
    )
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_kbath(data):

    t = data["time"] / 1000.0
    states = data["states"][:, :, 0, 0]

    K = states[:, 0]
    V = states[:, 2]

    nu = 100.0 * np.maximum(
        np.tanh((V - 25.0) / 20.0),
        0.0,
    )

    # Average into 1-second bins
    bins = np.floor(t).astype(int)
    valid = bins < 200

    counts = np.bincount(bins[valid], minlength=200)
    K_mean = np.bincount(
        bins[valid], weights=K[valid], minlength=200
    ) / counts
    nu_mean = np.bincount(
        bins[valid], weights=nu[valid], minlength=200
    ) / counts

    # Paper's fitted relationship
    K_grid = np.linspace(2.5, 20.0, 500)

    nu_paper = np.where(
        K_grid < 4.5,
        0.0,
        -63.9093
        + 20.0921 * K_grid
        - 1.53505 * K_grid**2
        + 0.0533615 * K_grid**3
        - 0.000690027 * K_grid**4,
    )

    plt.figure(figsize=(9, 5))
    plt.plot(K_grid, nu_paper, label="Chizhov et al. fit")
    plt.scatter(K_mean, nu_mean, s=12, alpha=0.65, label="TVB simulation")
    plt.xlabel("Extracellular potassium $K_o$ (mM)")
    plt.ylabel("Mean firing rate (Hz)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()