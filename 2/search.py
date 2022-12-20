""" This is a Python script to search for 100 initial points that lead to bad
local optima under regular gradient descent"""

from vqa import Z4
from pennylane import numpy as np
import pickle as pk

# OUTPUT FILEPATH
bad_file = "Z4/bad-points.pkl"
good_file = "Z4/good-points.pkl"

# CHOOSE VQA AND TARGET
vqa = Z4()
target = -3.99

# RUN SEARCH
bad_points = []
good_points = []
while len(bad_points) < 100:
    params = np.random.uniform(low=0, high=2 * np.pi, size=vqa.param_shape)
    _, result = vqa(params)
    if np.isclose(result, target, atol=0.1):
        print(f"Good: {result}")
        good_points.append(params)
    else:
        print(f"Bad: {result}")
        bad_points.append(params)

# WRITE TO FILE
with open(bad_file, 'wb') as fd:
    pk.dump(bad_points, fd)
with open(good_file, 'wb') as fd:
    pk.dump(good_points, fd)
