from vqa import MVC
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import qaoa
import pickle as pk

vqa = MVC(edges=[(0, 1), (1, 2), (2, 0), (2, 3)], nlayers=1)
dev = qml.device('default.qubit', wires=range(4))

cost_h, mixer_h = qaoa.min_vertex_cover(vqa.graph, constrained=False)


def qaoa_layer(gamma, alpha):
    qaoa.cost_layer(gamma, cost_h)
    qaoa.mixer_layer(alpha, mixer_h)


@qml.qnode(dev)
def probability_circuit(gamma, alpha):
    for i in range(4):
        qml.Hadamard(wires=i)
        qml.layer(qaoa_layer, 1, gamma, alpha)
    return qml.probs(wires=range(4))

r = 0.1
# filepath = f"(3) {r} params.pkl"
filepath = "good-points"
with open(filepath, "rb") as fd:
    param_mat = pk.load(fd)

for param in param_mat:
    print(vqa.cost(param))
    probs = probability_circuit(param[0], param[1])

    plt.style.use("seaborn")
    plt.bar(range(2 ** 4), probs)
    plt.show()
