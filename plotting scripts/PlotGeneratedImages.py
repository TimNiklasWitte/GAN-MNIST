from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns


def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)

    num_imgs_per_epoch = 5 #df.loc[0, "generated imgs"].shape[0]
    num_epochs = df.loc[:, "generated imgs"].shape[0]


    fig, axes = plt.subplots(5, 5)
    
    for idx in range(5):
        for epoch in range(5):
            img = df.loc[epoch*5, "generated imgs"]
            img = img[idx]
            axes[idx, epoch].imshow(img)

    plt.show() 
    # sns.lineplot(data=df.loc[:, ["discriminator fake loss"]], ax=axes[0], markers=True, legend=None)
    # axes[0].set_title("Discriminator fake loss")

    # sns.lineplot(data=df.loc[:, ["discriminator real loss"]], ax=axes[1], markers=True, legend=None)
    # axes[1].set_title("Discriminator real loss")

    # # grid
    # for ax in axes.flatten():
    #     ax.grid()

    # plt.tight_layout()
    # plt.savefig("../plots/DiscriminatorLosses.png")
    # plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")