""" This is a Python script to search for 100 initial points that lead to bad
local optima under regular gradient descent"""

from vqa import MVC
from pennylane import numpy as np
import pickle as pk

# OUTPUT FILEPATH
bad_file = "bad-points.pkl"
good_file = "good-points.pkl"

# CHOOSE VQA AND TARGET
vqa = MVC(edges=[(0, 1), (1, 2), (2, 0), (2, 3)], nlayers=1, max_iterations=None)
target = -0.05

# RUN SEARCH
bad_points = []
good_points = []
while len(bad_points) < 5:
    params = np.random.uniform(low=0, high=2 * np.pi, size=vqa.param_shape)
    initial_cost = vqa.cost(params)
    if initial_cost < target:
        print(f"Skip: {initial_cost}")
        continue
    _, result, _ = vqa(params)
    if result < target:
        print(f"Good: {result}")
        good_points.append(params)
    else:
        print(f"Bad: {result}")
        bad_points.append(params)

# WRITE TO FILE
with open(bad_file, 'wb') as fd:
    pk.dump(bad_points, fd)
# with open(good_file, 'wb') as fd:
#     pk.dump(good_points, fd)
