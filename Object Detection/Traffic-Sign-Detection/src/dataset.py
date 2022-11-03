import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from config import (
    RESIZE_TO, BATCH_SIZE, CLASSES, OUT_DIR,
    TRAIN_DIR_IMAGES, VALID_DIR_IMAGES,
    TRAIN_PATH_LABELS, VALID_PATH_LABELS,
)
from custom_utils import (
    collate_fn, visualize_sample,
    get_train_transform, get_valid_transform,
)

# the dataset class


class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_path, width, height, classes, transforms=None):
        self.width = width
        self.height = height
        self.classes = classes

        self.transforms = transforms
        self.images_dir = images_dir
        self.df_annots = pd.read_csv(labels_path)
        self.image_names = self.df_annots['file_name'].unique()

    def __getitem__(self, index):
        # capture the image name and the full image path
        image_name = self.image_names[index]
        records = self.df_annots[self.df_annots['file_name'] == image_name]

        # read the image
        image = cv2.imread(f"{self.images_dir}/{image_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # size ò original image
        image_width = image.shape[1]
        image_height = image.shape[0]

        # original bound box x_min, y_min, x_max, y_max format
        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values

        # resize bounding boxes
        boxes[:, 0] = (boxes[:, 0]/image_width)*self.width
        boxes[:, 1] = (boxes[:, 1]/image_height)*self.height
        boxes[:, 2] = (boxes[:, 2]/image_width)*self.width
        boxes[:, 3] = (boxes[:, 3]/image_height)*self.height

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # map the object name to `classes` list to get the label index
        labels = [self.classes.index(name) for name in records['class_name']]
        # lables to tensor
        labels = torch.tensor(labels, dtype=torch.int64)

        # supposing that all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['image_id'] = torch.tensor([index])

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return self.image_names.shape[0]


# prepare the final datasets and data loaders
def create_train_dataset():
    train_dataset = CustomDataset(
        TRAIN_DIR_IMAGES, TRAIN_PATH_LABELS,
        RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform()
    )
    return train_dataset


def create_valid_dataset():
    valid_dataset = CustomDataset(
        VALID_DIR_IMAGES, VALID_PATH_LABELS,
        RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform()
    )
    return valid_dataset


def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader


def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader


# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    train_dataset = CustomDataset(
        TRAIN_DIR_IMAGES, TRAIN_PATH_LABELS, RESIZE_TO, RESIZE_TO, CLASSES)
    print(f"Number of training images: {len(train_dataset)}")

    NUM_SAMPLES_TO_VISUALIZE = 3
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = train_dataset[i]
        save_dir = f"{OUT_DIR}/sample"
        visualize_sample(save_dir, image, target)
