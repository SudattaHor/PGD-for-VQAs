# Escaping saddle point in variational quantum algorithms with perturbed gradient descent

Read the report [here](./PGD-for-VQAs-report.pdf).

## Abstract
Variational quantum algorithms (VQAs), which
use classical algorithms to optimize parameter-
ized quantum circuits, are seen as the best hope
to achieve quantum advantage due to their wide
range of applications and compatibility with near-
term quantum computers. However, finding effi-
cient classical optimization strategies for VQAs
can be challenging. For example, the quantum ap-
proximate optimization algorithm (QAOA) has
loss landscapes that are generally non-convex,
and typical gradient descent optimization may
get stuck at bad local optima and saddle points. In
this work, we investigate the use of stochasticity
to escape saddle points. We provide evidence that
additional stochasticity can escape saddle points,
but may also lead to worse solutions

## Instructions for reproducing results

1. Navigate to `main/`.
2. Edit `vqa.py`, which contains classes for VQAs using Pennylane. The VQAs used in the report are already written, but you may change the optimizer, ansatz, and Hamiltonian however you choose. In the other files, call the correct VQA i.e. `vqa = MVC(...)`.
3. Run `search.py` to search for initial starting points that lead to suboptimal solutions.
4. Run `main.py index`, where index is an integer representing which initial point in `bad-points.pkl` to start from, to run the optimization over various noise levels (perturbation radii).
5. Run `plot-sol.py` to plot the probability distribution of the optimized ansatz.
6. Run `plot.py index` to plot loss vs. iterations, where index is an integer representing which initial point in `bad-points.pkl` to start from.


## Descriptions of folders 1-7

Folders 1-4 work with the VQA in section 4.1 in the report, while folders 5-7 work with the VQA in section 4.2.

- `1/` finds initial points that lead to suboptimal solutions and stores them in `bad_points.pkl`.
- `2/` is the same as `3/`. You can ignore this folder.
- `3/` optimizes the VQA for various noise levels (perturbation radii). Initial point is chosen so that no noise leads to a suboptimal solution. Analytically solves the circuit. 
- `4/` is the same as `3/`, except it uses a circuit with 1024 shots rather than an analytic solution of the circuit.
- `5/` finds initial points that lead to suboptimal solutions and stores them in `bad_points.pkl`. Conducts a grid search of the entire parameters space.
- `6/` optimizes the VQA for various noise levels (perturbation radii). Initial point is chosen so that no noise leads to a global minimum. Analytically solves the circuit.
- `7/` optimizes the VQA for various noise levels (perturbation radii). Initial points are chosen so that no noise leads to a saddle point. Analytically solves the circuit.
