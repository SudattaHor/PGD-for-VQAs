""" Script to run perturbed gradient descent on initial points that lead to
bad local optima in hopes of escaping them"""

from optimize import Optimize
import pennylane as qml
from pennylane import numpy as np
import pickle as pk

noise_level = 0.1

# INPUT FILEPATH
# TODO

# OUTPUT FILEPATH
# TODO

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
solver = Optimize(cost=cost, noise_lvl=noise_lvl)
shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=nqubits)
target = -3.99
bad_points = []
good_points = []
while len(bad_points) < 100:
    # TODO - INITIALIZE PARAM
    _, result = solver(params)
    if np.isclose(result, target, atol=0.1):
        print(f"Good: {result}")
        good_points.append(params)
    else:
        print(f"Bad: {result}")
        bad_points.append(params)

# WRITE TO FILE
with open(??, 'wb') as fd:
    pk.dump(bad_points, fd)
with open(??, 'wb') as fd:
    pk.dump(good_points, fd)
