""" Classes for each VQA"""

from pennylane import numpy as np
import pennylane as qml
import networkx as nx
from pennylane import qaoa


class MVC:
    """
    QAOA to solve the minimum vertex cover problem given a set of edges in a graph
    Pennylane
    """

    def __init__(self, edges, nlayers=2, max_iterations=100, step_size=0.01,
                 conv_tol=1e-05, noise_lvl=0, shots=None):

        # OPTIMIZATION PARAMETERS
        self.max_iterations = max_iterations
        self.conv_tol = conv_tol
        self.noise_lvl = noise_lvl
        self.opt = qml.GradientDescentOptimizer(step_size)

        # CIRCUIT PARAMETERS
        nqubits = 4

        # DEFINE GRAPH
        self.graph = nx.Graph(edges)

        # DEVICE
        # dev = qml.device('default.mixed', wires=nqubits, shots=1028)
        dev = qml.device('default.qubit', wires=nqubits, shots=shots)

        # HAMILTONIANS
        cost_h, mixer_h = qaoa.min_vertex_cover(self.graph, constrained=False)

        # ANSTAZ AND COST
        def qaoa_layer(gamma, alpha):
            qaoa.cost_layer(gamma, cost_h)
            qaoa.mixer_layer(alpha, mixer_h)

        @qml.qnode(dev)
        def cost(params):
            for i in range(nqubits):
                qml.Hadamard(wires=i)
            qml.layer(qaoa_layer, nlayers, params[0], params[1])
            return qml.expval(cost_h)
        self.cost = cost
        self.param_shape = (2, nlayers)

    def __call__(self, initial_params, ):
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

    def draw_graph(self, *args, **kwargs):
        nx.draw(self.graph, *args, **kwargs)
