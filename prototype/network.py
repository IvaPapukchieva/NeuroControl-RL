
import numpy as np
from tvb.simulator import coupling, integrators, monitors
from tvb.simulator.simulator import Simulator as TVBSimulator
from prototype.connectivity import conn, network_coupling
from prototype.epileptor import ChizhovEpileptor2

class CoupledChizhovEpileptor(ChizhovEpileptor2):
    """Add incoming voltage to the original model's voltage equation."""
    cvar = np.array([2], dtype=np.int32) 

    def dfun(self, state, coupling, local_coupling=0.0):
        derivatives = super().dfun(state, coupling, local_coupling)
        derivatives[2] += coupling[0] / self.tau_m
        return derivatives


def run_network(connection_rule, duration_ms=100.0):
    """Return monitor times and all four state variables for every region."""
    # in ms
    dt = 0.1  
    conn.set_idelays(dt)

    # Axes: history time, state variable, region, mode.
    initial_state = np.array([3.0, 10.0, 0.0, 1.0])  # K_o, Na_i, V, x_D
    history = np.broadcast_to( initial_state[None, :, None, None], (conn.horizon, 4, conn.number_of_regions, 1), ).copy()

    sim = TVBSimulator(
        model=CoupledChizhovEpileptor(),
        connectivity=conn,
        coupling=connection_rule,
        integrator=integrators.HeunDeterministic(dt=dt),
        monitors=(monitors.TemporalAverage(period=1.0),),
        initial_conditions=history,).configure()

    (time, activity), = sim.run(simulation_length=duration_ms)
    if activity.size == 0 or not np.all(np.isfinite(activity)):
        raise RuntimeError("Network produced empty or non-finite output")
    return time, activity


def main():
    print(f"Running our Chizhov Epileptor on {conn.number_of_regions} TVB regions...")
    _, uncoupled = run_network(coupling.Linear(a=np.array([0.0])))
    time, coupled = run_network(network_coupling)

    # The inherited monitor order is K_o, Na_i, V, x_D.
    voltage = coupled[:, 2, :, 0]
    difference = np.max(np.abs(voltage - uncoupled[:, 2, :, 0]))
    if difference <= 1e-8:
        raise RuntimeError("Coupling did not change the network voltage")

    print(f"Success: {len(time)} samples, {voltage.shape[1]} regions, all values finite.")
    print(f"Maximum voltage change with coupling enabled: {difference:.6g} mV")



if __name__ == "__main__":
    main()
