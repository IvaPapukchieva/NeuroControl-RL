import matplotlib.pyplot as plt


# Experiments (Fig. 6)

def plot_stationary_dynamics(time, lfp_values, rewards, 
action_time,frequencies_hz, stim_start, stim_end):

    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(11, 7),
        constrained_layout=True,
    )

    axes[0].plot(time, lfp_values, color="black", linewidth=0.6)
    axes[0].set_ylabel("LFP amplitude")

    axes[0].plot(
        [stim_start, stim_end], [1.05, 1.05],
        transform=axes[0].get_xaxis_transform(),
        color="black", linewidth=3, clip_on=False,
    )
    axes[0].text(
        (stim_start + stim_end) / 2, 1.08, "Stimulation",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
    )

    axes[1].plot(time, rewards, color="black", linewidth=1)
    axes[1].set_ylabel("Reward")

    axes[2].step(
        action_time, frequencies_hz,
        where="post", color="black",
    )
    axes[2].set_ylabel("Stimulus frequency (Hz)")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    return fig


# dummy data 
time_s = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
lfp_values = [0.2, -2.0, 0.3, -1.8, 0.2, 0.1, 0.05, 0.1, 0.2, -2.0, 0.3]
rewards = [0.8, -1.5, 0.7, -1.3, 0.8, 0.9, 1.0, 0.9, 0.8, -1.5, 0.7]
action_time_s = [0, 45, 60, 75, 90, 120, 150]
frequencies_hz = [0, 5, 3, 2, 2, 0, 0]

stim_start_s = 45
stim_end_s = 120

fig = plot_stationary_dynamics(
    time_s,
    lfp_values,
    rewards,
    action_time_s,
    frequencies_hz,
    stim_start_s,
    stim_end_s,)

plt.show()


