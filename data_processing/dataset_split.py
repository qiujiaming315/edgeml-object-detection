import argparse
import os
import numpy as np

"""Split the (validation) dataset of VOC/COCO into K folds for cross validation."""


def split_dataset(n_img, n_split, save_path, seed=0):
    """
    Split the dataset into n_split folds for cross validation.
    :param n_img: number of images in the dataset.
    :param n_split: number of splits.
    :param save_path: path of file to save the dataset split.
    :param seed: seed of random state generator.
    """
    # Check the inputs.
    assert n_split >= 1, "Please split the dataset into at least 2 folds."
    assert n_img >= n_split, "Please set a smaller number of splits."
    # Randomly shuffle the dataset.
    rstate = np.random.RandomState(seed)
    img_rank = np.arange(n_img)
    rstate.shuffle(img_rank)
    # Split the dataset according to the randomly shuffled order.
    split = np.zeros((n_split, n_img), dtype=bool)
    for split_idx in range(n_split):
        sub_data = img_rank[split_idx::n_split]
        split[split_idx, sub_data] = True
    # Save the dataset split.
    np.save(save_path, split)
    return


def main(opts):
    # Retrieve the number of images in the dataset.
    num_img = len(os.listdir(opts.img_dir))
    split_dataset(num_img, opts.num_split, opts.save_path)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('img_dir', help="Directory where the images in the (validation) dataset are stored.")
    args.add_argument('save_path', help="Path of file to save the dataset split.")
    args.add_argument('--num-split', type=int, default=5, help="The number of splits for cross validation.")
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
