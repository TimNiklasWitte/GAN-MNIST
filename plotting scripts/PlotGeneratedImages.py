from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns


def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)

    num_imgs_per_epoch = 5 #df.loc[0, "generated imgs"].shape[0]
    num_epochs = df.loc[:, "generated imgs"].shape[0] - 1


    fig, axes = plt.subplots(nrows=30, ncols=10, figsize=(5, 15))

    for epoch in range(num_epochs):
        imgs = df.loc[epoch, "generated imgs"]
        for idx_img in range(10):
            img = imgs[idx_img]
            axes[epoch, idx_img].imshow(img)
            axes[epoch, idx_img].axis("off")


    plt.tight_layout()
    plt.savefig("../plots/GeneratedImgsTraining.png", bbox_inches='tight')
    plt.show()
    
    # for epoch_idx in range(1,7):
    #     epoch = 5 * epoch_idx - 1

    #     if epoch > 30:
    #         epoch = 30

        
    #     imgs = df.loc[epoch, "generated imgs"]
        
    #     print(epoch, imgs.shape)

    #     for idx_img in range(num_imgs_per_epoch):
    #         img = imgs[idx_img]
            
    #         axes[idx_img, epoch_idx - 1].imshow(img)

        #print(imgs)
        # for epoch in range(6):
            
        #     print(epoch*5, img.shape)
            # img = img[idx]
            # axes[idx, epoch].imshow(img)

   
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