import matplotlib.pyplot as plt
import numpy as np


class OptLog():
    def __init__(self, init_param, init_cost):
        self.init_param = init_param
        self.init_cost = init_cost
        self.hist = np.array([[self.init_param, self.init_cost]])

    def update(self, param, cost):
        self.hist = np.append(self.hist, [[param, cost]], axis=0)

    def print(self):
        print("Param | Cost")
        print(self.hist)

    def plot(self):
        fig, [ax1, ax2] = plt.subplots(2)
        ax1.plot(range(self.hist.shape[0]), self.hist[:, 1])
        ax1.set_xlabel("Optimization step", fontsize=13)
        ax1.set_ylabel("Energy (Hartree)", fontsize=13)
        ax2.plot(range(self.hist.shape[0]), self.hist[:, 0])
        ax2.set_xlabel("Optimization step", fontsize=13)
        ax2.set_ylabel("Gate parameter $\\theta$ (rad)", fontsize=13)
        return fig


class OptimizerLog:
    """Log to store optimizer's intermediate results"""

    def __init__(self):
        self.loss = []

    def update(self, _nfevs, _theta, ftheta, *_):
        """Save intermediate results. Optimizers pass many values
        but we only store the third ."""
        self.loss.append(ftheta)
