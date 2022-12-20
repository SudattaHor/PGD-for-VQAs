""" This is a Python script to escape bad local optimal with PGD"""

from vqa import Z4
from pennylane import numpy as np
import pickle as pk
import sys

r = float(sys.argv[1])

# OUTPUT FILEPATH
output_file = f"{r}.pkl"

# INPUT FILEPATH
bad_file = "bad-points.pkl"

# OPEN DATA
with open(bad_file, 'rb') as fd:
    bad_points = pk.load(fd)
initial_point = bad_points[0]

# PARAMETERS
noise_lvl = r / np.sqrt(24)  # standard deviation of manually added noise
target = -2  # value of the bad local optimum
ntrials = 30

# OPTIMIZATION
vqa = Z4(noise_lvl=noise_lvl)
loss_curves = np.zeros((30, vqa.max_iterations))
for i in range(ntrials):
    hist, result_cost, _ = vqa(initial_point)
    loss_curves[i] = hist
    if result_cost < target:
        print(f"Good: {result_cost}")
    else:
        print(f"Bad: {result_cost}")

# SAVE
with open(output_file, 'wb') as fd:
    pk.dump(loss_curves, fd)
