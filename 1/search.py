""" This is a Python script to search for 100 initial points that lead to bad
local optima under regular gradient descent"""

from optimize import Optimize
import pennylane as qml
from pennylane import numpy as np
import pickle as pk

# OUTPUT FILEPATH
bad_file = "bad-points.pkl"
good_file = "good-points.pkl"

# DEVICE
nqubits = 4
dev = qml.device('default.qubit', wires=nqubits)

# HAMILTONIAN
coeffs = np.ones(nqubits)
obs = [qml.PauliZ(i) for i in range(nqubits)]
hamiltonian = qml.Hamiltonian(coeffs, obs)


# ANSTAZ AND COST
@qml.qnode(dev)
def cost(parameters):
    qml.StronglyEntanglingLayers(weights=parameters, wires=range(nqubits))
    return qml.expval(hamiltonian)


# SORT GOOD/BAD INITIAL POINTS
solver = Optimize(cost=cost)
target = -3.99
bad_points = []
good_points = []
while len(bad_points) < 100:
    params = np.random.uniform(low=0, high=2 * np.pi, size=shape)
    _, result = solver(params)
    if np.isclose(result, target, atol=0.1):
        print(f"Good: {result}")
        good_points.append(params)
    else:
        print(f"Bad: {result}")
        bad_points.append(params)

# WRITE TO FILE
with open(bad_file, 'wb') as fd:
    pk.dump(bad_points, fd)
with open(good_file, 'wb') as fd:
    pk.dump(good_points, fd)
