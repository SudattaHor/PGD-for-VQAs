import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle as pk
import numpy as np
from matplotlib import cm
from matplotlib import colors
import sys

keyword = str(sys.argv[1])

class MidpointNormalize(mpl.colors.Normalize):
    ## class from the mpl docs:
    # https://matplotlib.org/users/colormapnorms.html

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


with open("grid.pkl", "rb") as fd:
    grid = pk.load(fd)
with open(f"(3) {keyword} params.pkl", "rb") as fd:
    params = pk.load(fd)
bad_file = "bad-points100.pkl"
# OPEN DATA
with open(bad_file, 'rb') as fd:
    bad_points = pk.load(fd)
initial_point = bad_points[0]
for param in params:
    resolution = grid.shape
    target_param = np.unravel_index(np.argmin(grid, axis=None), grid.shape)
    print(target_param)
    X = np.linspace(0, np.pi, resolution[0])
    Y = np.linspace(0, np.pi, resolution[1])
    X, Y = np.meshgrid(X, Y)
    # fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "3d"})
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, extent=(0, 2 * np.pi, 0, 2 * np.pi), origin="lower", aspect="auto")
    #  norm=colors.Normalize(vmin=-1.4, vmax=0), cmap="coolwarm_r")
    # im = ax.imshow(grid, extent=(0, np.pi, 0, np.pi), origin="lower", aspect="auto", cmap="viridis")
    # im = ax.plot_surface(X, Y, grid, cmap=cm.viridis)
    x, y = param
    if y > 2*np.pi:
        y -= 2*np.pi
    ax.plot(x, y, marker='x', markersize=10, color='r')
    ax.plot(initial_point[0], initial_point[1], marker='o', color='r', markersize=10)
    xrange, yrange = np.linspace(0, 2*np.pi, resolution[0]), np.linspace(0, 2*np.pi, resolution[1])
    ax.plot(xrange[target_param[0]], yrange[target_param[1]], markersize=10, marker="*", color='r')
    ax.set_xlabel(r"$\gamma$", fontsize=20)
    ax.set_ylabel(r"$\alpha$    ", rotation=0, fontsize=20)
    clb = fig.colorbar(im)
    clb.ax.set_title("Cost", fontsize=20, rotation=0)
    # plt.savefig("grid.eps")
    plt.show()
    break
