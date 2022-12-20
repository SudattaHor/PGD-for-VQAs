""" Class to optimize VQAs in Pennylane"""

from pennylane import numpy as np
import pennylane as qml


class Optimize:
    def __init__(self, cost=None, max_iterations=100, step_size=0.5, conv_tol=1e-05, noise_lvl=0):
        self.cost = cost
        self.max_iterations = max_iterations
        self.conv_tol = conv_tol
        self.noise_lvl = noise_lvl
        self.opt = qml.GradientDescentOptimizer(step_size)

    def __call__(self, initial_params):
        params = initial_params
        cost_hist = []
        for n in range(self.max_iterations):
            params, prev_energy = self.opt.step_and_cost(self.cost, params)
            params += np.random.normal(loc=0, scale=self.noise_lvl, size=params.shape)
            cost_hist.append(prev_energy)
            energy = self.cost(params)
            conv = np.abs(energy - prev_energy)
            if conv <= self.conv_tol: break
        return cost_hist, energy.numpy()


class VQA:

    def __init__(self, max_iterations=100, step_size=0.5, conv_tol=1e-05, noise_lvl=0):

        self.max_iterations = max_iterations
        self.conv_tol = conv_tol
        self.noise_lvl = noise_lvl
        self.opt = qml.GradientDescentOptimizer(step_size)

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
        self.cost = cost

    def __call__(self, initial_params):
        params = initial_params
        cost_hist = []
        for _ in range(self.max_iterations):
            params, prev_energy = self.opt.step_and_cost(self.cost, params)
            params += np.random.normal(loc=0, scale=self.noise_lvl, size=params.shape)
            cost_hist.append(prev_energy)
            energy = self.cost(params)
            conv = np.abs(energy - prev_energy)
            if conv <= self.conv_tol: break
        return cost_hist, energy.numpy()
