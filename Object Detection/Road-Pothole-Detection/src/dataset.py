import torch
import numpy as np
import cv2
import glob
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from config import TRAIN_DIR, BATCH_SIZE, OUT_DIR
from utils import collate_fn, get_train_transform, get_valid_transform
from utils import visualize_sample, show_transformed_image

class PotHoleDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image_path = f"{self.image_dir}/Positive data/{image_id}.JPG"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
    
        # convert the boxes into x_min, y_min, x_max, y_max format
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # we have only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
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
            sample = self.transforms(image = image,
                                     bboxes = target['boxes'],
                                     labels = labels.tolist())
            image = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image, target

    def __len__(self):
        return self.image_ids.shape[0]

# read the annotation CSV file
train_df = pd.read_csv("../road-pothole-images-for-pothole-detection/train_df.csv")
print(train_df.head())
print(f"Total number of image IDs (objects) in dataframe: {len(train_df)}")
image_ids = train_df['image_id'].unique()
print(f"Total number of unique train images IDs in dataframe: {len(image_ids)}")

# get all the image paths as list
image_paths = glob.glob(f"{TRAIN_DIR}/Positive data/*.JPG")
image_names = [image_path.split('/')[-1].split('.')[0] for image_path in image_paths]
print(f"Total number of training images in folder: {len(image_names)}")

train_ids = image_names # use all the images for training
train_df = train_df[train_df['image_id'].isin(train_ids)]
print(f"Number of image IDs (objects) training on: {len(train_df)}")

# prepare the final datasets and data loaders
train_dataset = PotHoleDataset(train_df, TRAIN_DIR, get_train_transform())
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = PotHoleDataset(train_df, TRAIN_DIR)
    print(f"Number of training images: {len(dataset)}")
        
    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target, 'sample' + str(i))