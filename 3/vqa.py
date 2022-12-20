""" Classes for each VQA"""

from pennylane import numpy as np
import pennylane as qml


class Z4:
    """
    VQA with StronglyEntangling layers as the ansatz and Z1+Z2+Z3+Z4 as the Hamiltonian
    Pennylane
    """

    def __init__(self, max_iterations=100, step_size=0.5, conv_tol=1e-05, noise_lvl=0):

        self.max_iterations = max_iterations
        self.conv_tol = conv_tol
        self.noise_lvl = noise_lvl
        self.opt = qml.GradientDescentOptimizer(step_size)

        # DEVICE
        nqubits = 4
        dev = qml.device('default.mixed', wires=nqubits)
        # HAMILTONIAN
        coeffs = np.ones(nqubits)
        obs = [qml.PauliZ(i) for i in range(nqubits)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)

        # ANSTAZ AND COST
        @qml.qnode(dev)
        def cost(parameters):
            qml.StronglyEntanglingLayers(weights=parameters, wires=range(nqubits))
            return qml.expval(hamiltonian)
        self.cost = cost
        self.param_shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=nqubits)

    def __call__(self, initial_params):
        params = initial_params
        cost_hist = []
        for _ in range(self.max_iterations):
            params, prev_energy = self.opt.step_and_cost(self.cost, params)
            params += np.random.normal(loc=0, scale=self.noise_lvl, size=params.shape)
            cost_hist.append(prev_energy)
            energy = self.cost(params)
            # conv = np.abs(energy - prev_energy)
            # if conv <= self.conv_tol: break
        return cost_hist, energy, params
