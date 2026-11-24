
import numpy as np
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator import coupling

# an example connetivity file we can replace it with smth better
conn = Connectivity.from_file("connectivity_76.zip")
conn.speed = np.array([3.0])
conn.configure()

maximum_weight = conn.weights.max()
if maximum_weight <= 0:
    raise ValueError("Connectivity must contain positive connections")
conn.weights = conn.weights / maximum_weight

network_coupling = coupling.Linear(
    a=np.array([0.01]), 
    b=np.array([0.0]),
)
