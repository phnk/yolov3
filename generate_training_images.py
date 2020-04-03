from utils.utils import *
from torch.utils.data import DataLoader
from utils.datasets import *


if __name__ == "__main__":
    train_path = "/home/ph/Desktop/phd/data/scale model datasets/full dataset/train/train_paths.txt"
    dataset = LoadImagesAndLabels(train_path, img_size=1920, augment=False)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True, collate_fn=dataset.collate_fn, sampler=None)

    device = torch_utils.select_device()

    for i, (imgs, targets, _, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        if i < 10:
            plot_images(imgs=imgs, targets=targets, fname="train_img_{}.png".format(i))
        else:
            exit(1)