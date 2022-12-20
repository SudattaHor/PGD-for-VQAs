""" This is a Python script to escape bad local optimal with PGD"""

from vqa import MVC
from pennylane import numpy as np
import pickle as pk
import sys

r = float(sys.argv[1])

# OUTPUT FILEPATH
output_file = f"(3) {r}.pkl"
resulting_params_file = f"(3) {r} params.pkl"

# INPUT FILEPATH
bad_file = "bad-points100.pkl"

# OPEN DATA
with open(bad_file, 'rb') as fd:
    bad_points = pk.load(fd)
initial_point = bad_points[0]

# PARAMETERS
noise_lvl = r / np.sqrt(24)  # standard deviation of manually added noise
target = -1.35  # threshold for good optimum
ntrials = 30

# OPTIMIZATION
vqa = MVC(edges=[(0, 1), (1, 2), (2, 0), (2, 3)], nlayers=1, noise_lvl=noise_lvl, shots=None)
loss_curves = np.zeros((ntrials, vqa.max_iterations))
resulting_params = []
for i in range(ntrials):
    hist, result_cost, result_param = vqa(initial_point)
    loss_curves[i] = hist
    resulting_params.append(result_param)
    if result_cost < target:
        print(f"Good: {result_cost}")
    else:
        print(f"Bad: {result_cost}")

# SAVE
with open(output_file, 'wb') as fd:
    pk.dump(loss_curves, fd)
with open(resulting_params_file, 'wb') as fd:
    pk.dump(resulting_params, fd)
