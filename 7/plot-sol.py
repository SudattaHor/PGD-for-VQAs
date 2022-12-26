from vqa import MVC
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import qaoa
import pickle as pk

i = 0
r = 0.1
filename = f"bad_point[{i}]/final_param/{r}.pkl"
vqa = MVC(edges=[(0, 1), (1, 2), (2, 0), (2, 3)], nlayers=1)
with open(filename, "rb") as fd:
    data = pk.load(fd)
for param in data:
    print(vqa.cost(param))
    vqa.plot_prob(param)
