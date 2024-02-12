from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns


def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)

    fig, axes = plt.subplots(1, 2)
    
    sns.lineplot(data=df.loc[:, ["discriminator fake loss"]], ax=axes[0], markers=True, legend=None)
    axes[0].set_title("Discriminator fake loss")

    sns.lineplot(data=df.loc[:, ["discriminator real loss"]], ax=axes[1], markers=True, legend=None)
    axes[1].set_title("Discriminator real loss")

    # grid
    for ax in axes.flatten():
        ax.grid()

    plt.tight_layout()
    plt.savefig("../plots/DiscriminatorLosses.png")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")