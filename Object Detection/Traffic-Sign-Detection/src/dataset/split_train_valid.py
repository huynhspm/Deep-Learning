"""
Script to create a training and validation CSV file.
"""

import os
import shutil
import pandas as pd
from numpy import random


def train_valid_split(dir_path, split=0.2):
    all_images_dir = f"{dir_path}/TrainIJCNN2013"
    train_images_dir = f"{dir_path}/train_images"
    valid_images_dir = f"{dir_path}/valid_images"
    all_annots_path = f"{dir_path}/all_annots.csv"
    train_annots_path = f"{dir_path}/train_annots.csv"
    valid_annots_path = f"{dir_path}/valid_annots.csv"

    all_annots_df = pd.read_csv(all_annots_path)

    # get all unique images from csv-file
    all_images = all_annots_df['file_name'].unique()

    # shuffle images before split train-valid
    random.shuffle(all_images)

    len_all_images = len(all_images)
    split = int((1-split)*len_all_images)
    train_images = all_images[:split]
    valid_images = all_images[split:]

    train_annots_df = all_annots_df[all_annots_df['file_name'].isin(
        train_images)]
    valid_annots_df = all_annots_df[all_annots_df['file_name'].isin(
        valid_images)]

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)

    # Copy training images.
    for image in train_images:
        shutil.copy(
            f"{all_images_dir}/{image}",
            f"{train_images_dir}/{image}"
        )
    train_annots_df.to_csv(train_annots_path, index=False)

    # Copy validation images.
    for image in valid_images:
        shutil.copy(
            f"{all_images_dir}/{image}",
            f"{valid_images_dir}/{image}"
        )
    valid_annots_df.to_csv(valid_annots_path, index=False)


train_valid_split(dir_path="../../input")
