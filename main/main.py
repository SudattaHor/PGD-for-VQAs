""" This is a Python script to escape bad local optimal with PGD"""

from vqa import MVC
from pennylane import numpy as np
import pickle as pk
import sys

bad_point_index = int(sys.argv[1])

for r in [0, 0.1]:
    # OUTPUT FILEPATH
    output_file = f"bad_point[{bad_point_index}]/loss_curves/{r}.pkl"
    param_file = f"bad_point[{bad_point_index}]/final_param/{r}.pkl"

    # INPUT FILEPATH
    bad_file = "bad-points.pkl"

    # OPEN DATA
    with open(bad_file, 'rb') as fd:
        bad_points = pk.load(fd)
    initial_point = bad_points[bad_point_index]

    # PARAMETERS
    noise_lvl = r / np.sqrt(24)  # standard deviation of manually added noise
    target = -1.35  # threshold for good optimum
    ntrials = 30

    # OPTIMIZATION
    vqa = MVC(edges=[(0, 1), (1, 2), (2, 0), (2, 3)], nlayers=1, noise_lvl=noise_lvl, max_iterations=400)
    loss_curves = np.zeros((ntrials, vqa.max_iterations))
    final_params = []
    for i in range(ntrials):
        hist, result_cost, final_param = vqa(initial_point)
        loss_curves[i] = hist
        final_params.append(final_param)
        if result_cost < target:
            print(f"Good: {result_cost}")
        else:
            print(f"Bad: {result_cost}")

    # SAVE
    with open(output_file, 'wb') as fd:
        pk.dump(loss_curves, fd)
    with open(param_file, "wb") as fd:
        pk.dump(final_params, fd)
