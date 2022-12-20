import matplotlib.pyplot as plt
import pickle as pk
import numpy as np

# FILEPATHS
keywords = [f"{i:2.1f}" for i in np.linspace(0, 0.5, 6)]
for keyword in keywords:
    input_filepath = f"{keyword}.pkl"
    output_filepath = f"{keyword}.eps"

    with open(input_filepath, "rb") as fd:
        loss_curves = pk.load(fd)

    # PLOTTING
    ngood = 0
    btarget = -1.35
    gtarget = -1.41
    fig, ax = plt.subplots(figsize=(10, 8))
    for curve in loss_curves:
        if curve[-1] < btarget:
            ngood += 1
        ax.plot(curve)
    print(f"key: {keyword} | ngood: {ngood} | ntotal: {len(loss_curves)}")
    plt.axhline(y=gtarget, color='r', linestyle='--', label="Target", linewidth=2)
    plt.ylim([gtarget - 0.25, curve[0] + 0.25])
    plt.xlabel("Iterations", fontsize=20)
    plt.ylabel("Loss", fontsize=20, rotation=0, loc="top")
    plt.legend(fontsize=20)
    plt.title(f"r = {keyword}", fontsize=20)
    plt.savefig(output_filepath)
    plt.show()
