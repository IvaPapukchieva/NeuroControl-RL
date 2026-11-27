"""
Potassium estimator
Estimating the inner state variable K0 from the observable U.

"""

import numpy as np

from prototype.epileptor import ChizhovEpileptor2
from prototype.observer import ChizhovObserver


def simulate(duration_s=600.0, seed=0, noise_mV=25.0, gL_nS=1.0, dt=1.0, **model_params):
    """Euler-Maruyama run of ChizhovEpileptor2 to create data for  training the model (for that we need both U and true K from the inner state)
    """
    model = ChizhovEpileptor2()
    for name, value in model_params.items():
        setattr(model, name, np.array([value]))
    model.configure()

    tau_m = float(model.tau_m[0])
    observer = ChizhovObserver(gL_nS)
    rng = np.random.default_rng(seed)

    state = np.array([[[3.0]], [[10.0]], [[0.0]], [[1.0]]])  # K_o, Na_i, V, x_D
    coupling = np.zeros((1, 1, 1))

    n = int(duration_s * 1000 / dt)
    U = np.empty(n)
    spikes = np.empty(n, dtype=bool)
    K = np.empty(n)

    for i in range(n):
        new = state + dt * model.dfun(state, coupling)
        new[2] += noise_mV / tau_m * np.sqrt(dt) * rng.standard_normal()
        U[i], spikes[i] = observer.step(state, new, dt, tau_m)
        state = new
        K[i] = state[0, 0, 0]

    return {"U": U, "spikes": spikes, "K": K, "dt": dt}


class PotassiumEstimator:
    """Estimates K0 from U (and its spikes) only.
    """

    FEATURE_NAMES = [
        "bias", "U_median", "U_std", "log_rate",
        "ema10_median", "ema60_median", "ema10_lograte", "ema60_lograte",
    ]

    def __init__(self, dt=1.0, window_ms=1000.0, weights=None,
                 bins=(3.6, 4.2), ictal_rate_hz=0.5):
        self.dt = dt
        self.window_n = int(window_ms / dt)
        self.window_s = window_ms / 1000.0 #one estimate  per second
        self.weights = weights
        self.bins = bins #thresholds for different states depending on K (for now handcrafted)
        self.ictal_rate_hz = ictal_rate_hz #threshold for the seizure based on K
        self.reset() #called after computing the features every second

    def reset(self):
        self._buf = []
        self._spk = 0
        self._ema = None
        self.features = None
        self.rate_hz = 0.0
        self.K_hat = np.nan

    def _window_features(self):
        """
        - median: more K outside means more depolarized, so U higher (but median to ignore  spike rates)
        - std: measturing the deviation, if no seizure then U more stable
        - log_rate: spike rate means a burst or a seizure and through log we make the seizure less dominating
        - ema10 and ema60 are the slow running avearges (K changes slowly) of the median and the spike rate so that we incorporate memory data
        """
        u = np.asarray(self._buf)
        median = np.median(u)
        std = u.std()
        log_rate = np.log1p(self._spk / self.window_s)
        self.rate_hz = self._spk / self.window_s

        x = np.array([median, median, log_rate, log_rate])
        if self._ema is None:
            self._ema = x.copy()
        a = np.array([self.window_s / 10, self.window_s / 60,
                      self.window_s / 10, self.window_s / 60])
        self._ema += a * (x - self._ema)

        return np.array([1.0, median, std, log_rate, *self._ema])

    def push(self, U, spiked=False):
        """returns a new K estimate at the end of each window and otherwise None"""
        self._buf.append(U)
        self._spk += int(spiked)
        if len(self._buf) < self.window_n:
            return None

        self.features = self._window_features()
        self._buf = []
        self._spk = 0
        if self.weights is not None:
            self.K_hat = float(self.features @ self.weights)
        return self.features

    def get_state(self):
        """
        0 = K low, 1 = K medium, 2 = K high (close to seizure), 3 = seizing"""
        if self.rate_hz > self.ictal_rate_hz:
            return len(self.bins) + 1
        return int(np.searchsorted(self.bins, self.K_hat))


    #some helpers for the training
    def run_offline(self, U, spikes, K=None):
        """replays a recorded trace, returns features (and window-averaged true K)"""
        self.reset()
        feats, k_true = [], []
        for i in range(len(U)):
            f = self.push(U[i], spikes[i])
            if f is not None:
                feats.append(f)
                if K is not None:
                    k_true.append(K[i - self.window_n + 1:i + 1].mean())
        return np.array(feats), np.array(k_true)

    def fit(self, runs, ridge=1e-3, warmup_s=30):
        """Simple ridge regression features, tries to predict K on the training runs
        later possibly improved to a small tree or small neural-net
        """
        X, y = [], []
        skip = int(warmup_s / self.window_s)
        for r in runs:
            f, k = self.run_offline(r["U"], r["spikes"], r["K"])
            X.append(f[skip:])
            y.append(k[skip:])
        X, y = np.vstack(X), np.concatenate(y)
        mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-12 #standardising
        Z = np.hstack([np.ones((len(X), 1)), (X[:, 1:] - mu) / sd])
        w = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ y)
        self.weights = np.concatenate([[w[0] - np.sum(w[1:] * mu / sd)], w[1:] / sd])
        return self.weights

    def evaluate(self, run, warmup_s=30):
        f, k = self.run_offline(run["U"], run["spikes"], run["K"])
        skip = int(warmup_s / self.window_s)
        f, k = f[skip:], k[skip:]
        k_hat = f @ self.weights
        r2 = 1 - np.sum((k - k_hat) ** 2) / np.sum((k - k.mean()) ** 2)
        mae = np.mean(np.abs(k - k_hat))
        return {"R2": r2, "MAE_mM": mae, "K_true": k, "K_hat": k_hat}

    def save(self, path):
        np.savez(path, weights=self.weights, bins=np.array(self.bins))

    @classmethod
    def load(cls, path, **kwargs):
        d = np.load(path)
        return cls(weights=d["weights"], bins=tuple(d["bins"]), **kwargs)



#Training on two seeds (600s each) and run on a third (new) seed
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = 600.0
    print("Simulating training runs ...")
    train = [simulate(T, seed=s) for s in (1, 2)]

    est = PotassiumEstimator()
    est.fit(train)
    est.save("potassium_estimator_weights.npz")
    print("weights:", dict(zip(PotassiumEstimator.FEATURE_NAMES, np.round(est.weights, 4))))

    tests = {
        "unseen seed": simulate(T, seed=10),
        "unseen seed, noise 20 mV": simulate(T, seed=11, noise_mV=20.0),
        "unseen seed, K_bath 8.0": simulate(T, seed=12, K_bath=8.0),
    }

    fig, axes = plt.subplots(len(tests), 1, figsize=(11, 2.8 * len(tests)), sharex=True)
    for ax, (name, run) in zip(axes, tests.items()):
        res = est.evaluate(run)
        print(f"{name:28s} R2 = {res['R2']:.3f}   MAE = {res['MAE_mM']:.3f} mM")
        t = 30 + np.arange(len(res["K_true"])) * est.window_s
        ax.plot(t, res["K_true"], label="true K (hidden)", lw=1.5)
        ax.plot(t, res["K_hat"], label="estimated K (from U only)", lw=1)
        for b in est.bins:
            ax.axhline(b, color="gray", ls=":", lw=0.8)
        ax.set_title(f"{name}:  R² = {res['R2']:.2f}, MAE = {res['MAE_mM']:.2f} mM")
        ax.set_ylabel("K_o (mM)")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig("potassium_estimate.png", dpi=120)
    print("saved potassium_estimate.png")
