""" This is a Python script to use grid-search to optimize QAOA"""

from vqa import MVC
from pennylane import numpy as np
import pickle as pk

# OUTPUT FILEPATH
output_filename = "grid2.pkl"

resolution = (1000, 1000)
vqa = MVC(edges=[(0, 1), (1, 2), (2, 0), (2, 3)], nlayers=1)

grid = np.zeros(resolution)
gammas = np.linspace(0, np.pi, resolution[0])
alphas = np.linspace(0, np.pi, resolution[1])
for i, gamma in enumerate(gammas):
    for j, alpha in enumerate(alphas):
        print(f"Computing i: {i}, j: {j}")
        grid[i][j] = vqa.cost(np.array([[gamma], [alpha]]))

# WRITE TO FILE
with open(output_filename, 'wb') as fd:
    pk.dump(grid, fd)
