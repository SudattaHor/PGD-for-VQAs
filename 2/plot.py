import matplotlib.pyplot as plt
import pickle as pk

# FILEPATHS
keywords = ["0", "1e-1", "2e-1", "5e-1", "1e1"]
for keyword in keywords:
    input_filepath = f"{keyword}.pkl"
    output_filepath = f"{keyword}.png"

    with open(input_filepath, "rb") as fd:
        loss_curves = pk.load(fd)

    # PLOTTING
    ngood = 0
    btarget = -2
    gtarget = -4
    fig, ax = plt.subplots(figsize=(10, 8))
    for curve in loss_curves:
        if curve[-1] < btarget:
            ngood += 1
        ax.plot(curve)
    print(f"ngood: {ngood} | ntotal: {len(loss_curves)}")
    plt.axhline(y=gtarget, color='r', linestyle='--', label="Target", linewidth=2)
    plt.ylim([-4.25, -0.25])
    plt.xlabel("Iterations", fontsize=20)
    plt.ylabel("Loss", fontsize=20, rotation=0, loc="top")
    plt.legend(fontsize=20)
    plt.savefig(output_filepath)
    plt.show()
