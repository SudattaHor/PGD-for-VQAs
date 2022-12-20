from vqa import MVC
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import qaoa
import pickle as pk

vqa = MVC(edges=[(0, 1), (1, 2), (2, 0), (2, 3)], nlayers=1)
param = [[0.32057067893773394], [2.756907838864512]]
print(vqa.cost(param))
vqa.plot_prob(param)
