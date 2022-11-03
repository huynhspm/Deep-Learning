import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from config import NUM_CLASSES, CLASSES, OUT_DIR

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(NUM_CLASSES, 3))
def draw_boxes(boxes, labels, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, CLASSES[labels[i]], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, 
                    lineType=cv2.LINE_AA)
    return image

# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

# define the training tranforms
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        # A.RandomRotate90(0.5),
        # A.MotionBlur(p=0.2),
        # A.MedianBlur(blur_limit=3, p=0.1),
        # A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })

# function to visualize a single sample
def visualize_sample(image, target, name_image):
    boxes = target['boxes']
    labels = target['labels']   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     
    image = draw_boxes(boxes, labels, image * 255.0)
    cv2.imwrite(f"{OUT_DIR}/{name_image}.jpg", image)
    cv2.waitKey(0)

def show_transformed_image(dataloader):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    """
    if len(dataloader) > 0:
        images, targets = next(iter(dataloader))
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        for i in range(1):
            print(images[i].shape)
            sample = images[i].permute(1,2,0).cpu().numpy()
            visualize_sample(sample, targets[i], 'tranformed_image' + str(i))