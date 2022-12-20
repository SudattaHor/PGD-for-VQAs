import pennylane as qml
from pennylane import numpy as np
from util.opt_log import OptLog
import matplotlib.pyplot as plt

# INITIALIZE H2
symbols = ['H', 'H']
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
electrons = 2

# INITIALIZE DEVICE
dev = qml.device("default.qubit", wires=qubits)
hf = qml.qchem.hf_state(electrons, qubits)  # Hartree-Fock


def circuit(param, wires):
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])


@qml.qnode(dev)
def cost_fn(param):
    circuit(param, wires=range(qubits))
    return qml.expval(H)


opt = qml.GradientDescentOptimizer(stepsize=0.4)


theta = 0
length = 2*np.pi
max_iterations = 25

log = OptLog(theta, cost_fn(theta))
for theta in np.linspace(theta, length, max_iterations)[1:]:
    log.update(theta, cost_fn(theta))
    energy = log.hist[:, 1]
    print(f"Theta = {theta},  Energy = {energy[-1]:.8f} Ha")

plt.plot(log.hist[:, 0], log.hist[:, 1], '-o', linewidth=3)
plt.xlabel(r"$\theta$ (Radians)", size=12)
plt.ylabel("Energy", size=12)
plt.show()
