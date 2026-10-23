import numpy as np
import matplotlib.pyplot as plt

def plot_state_space(data):
    plt.figure(figsize=(8, 6))
    plt.plot(data["x1"], data["x2"], linewidth=1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("State-Space Trajectory")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_seizure_trajectory(data, z_shift=5.0, ax=None):
    """
    Plot the seizure trajectory in 3D.

    X = x1
    Y = x2
    Z = z + z_shift

    Parameters
    ----------
    data : dict
        Simulation output containing 'x1', 'x2', and 'z'.
    z_shift : float, default=5.0
        Upward shift applied to z for plotting.
    ax : matplotlib.axes.Axes, optional
        Existing 3D axes to plot on.

    Returns
    -------
    fig, ax
        Figure and 3D axes.
    """
    x1 = np.asarray(data["x1"]).ravel()
    x2 = np.asarray(data["x2"]).ravel()
    z = np.asarray(data["z"]).ravel() + z_shift

    if not (len(x1) == len(x2) == len(z)):
        raise ValueError("x1, x2, and z must have the same number of samples.")

    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.plot(-x1, x2, z, linewidth=0.8)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$z$ (shifted)")
    ax.set_title("3D Seizure Trajectory")

    fig.tight_layout()
    plt.show()

def plot_seizure_signals(data, figsize=(12, 8)):
    """
    Plot LFP, smoothed 2–15 Hz power, and slow variable z over time.

    Parameters
    ----------
    data : dict
        Simulation data containing 'time', 'lfp', 'z'.
        Optionally contains 'low' and 'high'.
    figsize : tuple
        Figure size.

    """
    time = np.asarray(data["time"]).ravel()

    signals = [
        ("lfp", "LFP"),
        ("last_power", "Smoothed 2–15 Hz power"),
        ("z", "Slow variable z"),
    ]

    fig, axes = plt.subplots(
        len(signals), 1, figsize=figsize, sharex=True
    )

    for ax, (key, label) in zip(axes, signals):
        if key not in data:
            ax.text(
                0.5, 0.5, f"'{key}' not present in data",
                ha="center", va="center", transform=ax.transAxes
            )
            ax.set_ylabel(label)
            continue

        values = np.asarray(data[key]).ravel()
        if len(values) != len(time):
            raise ValueError(
                f"'{key}' has {len(values)} samples, "
                f"but time has {len(time)}."
            )

        ax.plot(time, values, linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    plt.show()