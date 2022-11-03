import torch

BACKBONE = 'mobilenet_v3_large_320' # 'mobilenet_v3_large'  ;  'resnet50'
BATCH_SIZE = 4  # increase / decrease according to GPU memeory
RESIZE_TO = 512  # resize the image for training and transforms
NUM_EPOCHS = 200  # number of epochs to train for

# If you are on any of the Linux systems and trying to execute 
# the code locally, try using a value of 2 or above. 
# This will likely decrease the data fetching time.
NUM_WORKERS = 0 

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# Images and labels direcotry should be relative to train.py
TRAIN_DIR_IMAGES = "../input/train_images"
TRAIN_PATH_LABELS = "../input/train_annots.csv"
VALID_DIR_IMAGES = "../input/valid_images"
VALID_PATH_LABELS = "../input/valid_annots.csv"

# location to save model and plots
OUT_DIR = "../output"

# classes: 0 index is reserved for background
CLASSES = [
    '__background__',
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'End of speed limit (80km/h)', 'Speed limit (100km/h)',
    'Speed limit (120km/h)', 'No passing',
    'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield',
    'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
    'No entry', 'General caution', 'Dangerous curve to the left',
    'Dangerous curve to the right', 'Double curve', 'Bumpy road',
    'Slippery road', 'Road narrows on the right', 'Road work',
    'Traffic signals', 'Pedestrians', 'Children crossing',
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End of all speed and passing limits', 'Turn right ahead',
    'Turn left ahead', 'Ahead only', 'Go straight or right',
    'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
    'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
]
NUM_CLASSES = len(CLASSES)

MIN_SIZE = 800
# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

PREDICTION_THRES = 0.3

TEST_DIR = "../input/TestIJCNN2013"
INFERENCE_DIR = "../inference"

SAVE_PLOTS_EPOCH = 10  # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 10  # save model after these many epochs
