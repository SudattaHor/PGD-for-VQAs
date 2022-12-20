import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle as pk
import numpy as np
from matplotlib import cm
from matplotlib import colors


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


i = 0
r = 0.1
size = 100

with open("grid.pkl", "rb") as fd:
    grid = pk.load(fd)
with open("bad-points.pkl", "rb") as fd:
    bad_points = pk.load(fd)
with open(f"bad_point[{i}]/final_param/{r}.pkl", "rb") as fd:
    final_params = pk.load(fd)
bad_point = bad_points[i]
print(bad_points)
ngrid = grid
target_param = np.unravel_index(np.argmin(ngrid, axis=None), ngrid.shape)
resolution = grid.shape
X = np.linspace(0, np.pi, resolution[0])
Y = np.linspace(0, np.pi, resolution[1])
X, Y = np.meshgrid(X, Y)
for x, y in final_params:
    # fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "3d"})
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, extent=(0, 2 * np.pi, 0, 2 * np.pi), origin="lower", aspect="auto")
    #  norm=colors.Normalize(vmin=-1.4, vmax=0), cmap="coolwarm_r")
    # im = ax.imshow(grid, extent=(0, np.pi, 0, np.pi), origin="lower", aspect="auto", cmap="viridis")
    # im = ax.plot_surface(X, Y, grid, cmap=cm.viridis)
    xrange, yrange = np.linspace(0, 2*np.pi, 100), np.linspace(0, 2*np.pi, 100)
    ax.scatter(xrange[target_param[0]], yrange[target_param[1]], s=size, marker="*", color='r', label="Target")
    print((xrange[target_param[0]], yrange[target_param[1]]))
    bad_point_x, bad_point_y = bad_point
    ax.scatter(bad_point_x, bad_point_y, marker='o', s=size, color='r', label="Initial Point")
    ax.scatter(x, y, marker="x", s=size, color='r', label="Result")
    ax.set_xlabel(r"$\gamma$", fontsize=20)
    ax.set_ylabel(r"$\alpha$    ", rotation=0, fontsize=20)
    # ax.set_xlim([0, np.pi])
    # ax.set_ylim([0, 4])
    clb = fig.colorbar(im)
    clb.ax.set_title("Cost", fontsize=20, rotation=0)
    ax.legend()
    # plt.savefig("grid.eps")
    plt.show()
    break
